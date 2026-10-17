# Build and push once, then create a RunPod serverless endpoint:
#   docker build -t YOUR_DOCKERHUB/art-lab-tooncrafter:latest workers/tooncrafter
#   docker push YOUR_DOCKERHUB/art-lab-tooncrafter:latest
# Endpoint: 24 GB GPU, min workers 0, network volume mounted at /runpod-volume.
# Put the endpoint id in .env as RUNPOD_ENDPOINT_ID.
# The first job downloads the ToonCrafter fp16 weights and the Real-ESRGAN
# anime-video weights onto that volume. Later jobs reuse them.

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Doubiiu/ToonCrafter.git /opt/tooncrafter \
    && git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /opt/Real-ESRGAN

WORKDIR /opt/tooncrafter

COPY requirements.txt /tmp/worker-requirements.txt

# Dependency installs can replace the base image's CUDA torch. Put that torch back
# before xformers so the diffusion model still sees the GPU.
RUN pip install --no-cache-dir -r /tmp/worker-requirements.txt \
    && pip install --no-cache-dir realesrgan basicsr facexlib gfpgan \
    && pip install --no-cache-dir --force-reinstall \
        torch==2.2.1 torchvision==0.17.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir --no-deps xformers==0.0.25 \
    && pip install --no-cache-dir "numpy>=1.24,<2" \
    && python -c "import pathlib; matches = list(pathlib.Path('/usr/local/lib').rglob('basicsr/data/degradations.py')); assert len(matches) == 1, matches; p = matches[0]; p.write_text(p.read_text().replace('torchvision.transforms.functional_tensor', 'torchvision.transforms.functional'))"

COPY handler.py engine.py /opt/tooncrafter/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "/opt/tooncrafter/handler.py"]
