import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from safetensors.torch import load_file

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

CKPT_NAME = "tooncrafter_512_interp-pruned-fp16.safetensors"
CKPT_REPO = "Kijai/DynamiCrafter_pruned"
ESRGAN_NAME = "realesr-animevideov3.pth"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
RESOLUTION = (320, 512)

_model = None


def weights_dir():
    configured = os.environ.get("WEIGHTS_DIR", "").strip()
    if configured:
        path = Path(configured)
    elif Path("/runpod-volume").is_dir():
        path = Path("/runpod-volume")
    else:
        path = Path("/opt/weights")
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(path / "hf")
    os.environ["TORCH_HOME"] = str(path / "torch")
    return path


def ensure_checkpoint():
    cache = weights_dir()
    dest = cache / CKPT_NAME
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"Downloading {CKPT_NAME} onto {cache}", flush=True)
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=CKPT_REPO,
                filename=CKPT_NAME,
                local_dir=str(cache),
                local_dir_use_symlinks=False,
            )
        )
    except TypeError:
        downloaded = Path(
            hf_hub_download(
                repo_id=CKPT_REPO,
                filename=CKPT_NAME,
                local_dir=str(cache),
            )
        )
    if downloaded != dest and downloaded.exists():
        downloaded.replace(dest)
    return dest


def ensure_upscaler():
    cache = weights_dir()
    dest = cache / ESRGAN_NAME
    if not dest.exists() or dest.stat().st_size == 0:
        import requests

        print(f"Downloading {ESRGAN_NAME} onto {cache}", flush=True)
        response = requests.get(ESRGAN_URL, stream=True, timeout=120)
        response.raise_for_status()
        partial = dest.with_suffix(".part")
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        partial.replace(dest)
    link_dir = Path("/opt/Real-ESRGAN/weights")
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / ESRGAN_NAME
    if not link.exists():
        link.symlink_to(dest)
    return dest


def release_model():
    global _model
    if _model is None:
        return
    _model.cpu()
    del _model
    _model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _filtered_state(path):
    state = load_file(str(path), device="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        if name.startswith("cond_stage_model") or name.startswith("embedder"):
            continue
        cleaned[name] = value
    return cleaned


def get_model():
    global _model
    if _model is not None:
        return _model
    if not torch.cuda.is_available():
        raise RuntimeError("ToonCrafter needs a CUDA GPU.")
    os.chdir(APP_ROOT)
    checkpoint = ensure_checkpoint()
    from utils.utils import instantiate_from_config

    config = OmegaConf.load(APP_ROOT / "configs" / "inference_512_v1.0.yaml")
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model = instantiate_from_config(model_config)
    missing, unexpected = model.load_state_dict(_filtered_state(checkpoint), strict=False)
    bad_missing = [
        key
        for key in missing
        if not (key.startswith("cond_stage_model") or key.startswith("embedder"))
    ]
    if bad_missing:
        preview = ", ".join(bad_missing[:6])
        raise RuntimeError(f"The ToonCrafter checkpoint is missing weights: {preview}")
    if unexpected:
        preview = ", ".join(unexpected[:6])
        raise RuntimeError(f"The ToonCrafter checkpoint has unexpected weights: {preview}")
    model = model.half().cuda()
    model.eval()
    _model = model
    print("ToonCrafter checkpoint is loaded.", flush=True)
    return model


def _as_numpy(image):
    if isinstance(image, np.ndarray):
        array = image
    else:
        array = np.asarray(Image.open(image).convert("RGB"))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        array = np.asarray(Image.fromarray(array).convert("RGB"))
    return array


def _write_mp4(batch_samples, dest, fps):
    import imageio.v2 as imageio

    video = batch_samples[0, 0].detach().float().cpu()
    video = torch.clamp(video, -1, 1)
    frames = ((video + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()
    options = {
        "fps": fps,
        "codec": "libx264",
        "quality": 8,
        "macro_block_size": 1,
    }
    try:
        imageio.mimwrite(dest, frames, ffmpeg_params=["-pix_fmt", "yuv420p"], **options)
    except TypeError:
        imageio.mimwrite(dest, frames, output_params=["-pix_fmt", "yuv420p"], **options)


def interpolate(image1, image2, prompt, steps, cfg_scale, eta, frame_stride, seed, dest):
    from scripts.evaluation.funcs import batch_ddim_sampling

    import torchvision.transforms as transforms

    seed_everything(int(seed))
    model = get_model()
    steps = min(60, max(1, int(steps)))
    transform = transforms.Compose(
        [
            transforms.Resize(min(RESOLUTION)),
            transforms.CenterCrop(RESOLUTION),
        ]
    )
    torch.cuda.empty_cache()
    frames = model.temporal_length
    channels = model.model.diffusion_model.out_channels
    height, width = RESOLUTION[0] // 8, RESOLUTION[1] // 8
    noise_shape = [1, channels, frames, height, width]
    image = _as_numpy(image1)
    image_end = _as_numpy(image2)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_emb = model.get_learned_conditioning([prompt])
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
        img_tensor = (img_tensor / 255.0 - 0.5) * 2
        videos = transform(img_tensor).unsqueeze(0).unsqueeze(2)
        videos = repeat(videos, "b c t h w -> b c (repeat t) h w", repeat=frames // 2)

        img_tensor2 = torch.from_numpy(image_end).permute(2, 0, 1).float().to(model.device)
        img_tensor2 = (img_tensor2 / 255.0 - 0.5) * 2
        videos2 = transform(img_tensor2).unsqueeze(0).unsqueeze(2)
        videos2 = repeat(videos2, "b c t h w -> b c (repeat t) h w", repeat=frames // 2)
        videos = torch.cat([videos, videos2], dim=2)
        latent, hidden = _latent_with_hidden(model, videos)

        image_condition = torch.zeros_like(latent)
        image_condition[:, :, :1, :, :] = latent[:, :, :1, :, :]
        image_condition[:, :, -1:, :, :] = latent[:, :, -1:, :, :]

        cond_images = model.embedder(img_tensor.unsqueeze(0))
        image_emb = model.image_proj_model(cond_images)
        imtext_cond = torch.cat([text_emb, image_emb], dim=1)
        frame_stride_tensor = torch.tensor([int(frame_stride)], dtype=torch.long, device=model.device)
        cond = {
            "c_crossattn": [imtext_cond],
            "fs": frame_stride_tensor,
            "c_concat": [image_condition],
        }
        samples = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            n_samples=1,
            ddim_steps=steps,
            ddim_eta=float(eta),
            cfg_scale=float(cfg_scale),
            hs=hidden,
        )

    _write_mp4(samples, dest, fps=8)
    return dest


def _latent_with_hidden(model, videos):
    from einops import rearrange

    batch, _channels, time, _height, _width = videos.shape
    flat = rearrange(videos, "b c t h w -> (b t) c h w")
    encoder_posterior, hidden_states = model.first_stage_model.encode(flat, return_hidden_states=True)
    hidden_first_last = []
    for hidden in hidden_states:
        hidden = rearrange(hidden, "(b t) c h w -> b c t h w", t=time)
        hidden_first_last.append(torch.cat([hidden[:, :, 0:1], hidden[:, :, -1:]], dim=2))
    latent = model.get_first_stage_encoding(encoder_posterior).detach()
    latent = rearrange(latent, "(b t) c h w -> b c t h w", b=batch, t=time)
    return latent, hidden_first_last
