# Hugging Face Spaces (Docker SDK) image for the public demo.
# Builds straight from the GitHub repo, so the Space only needs this file and README.md.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Spaces run the container as uid 1000, which must own every folder the app writes to.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH YOLO_CONFIG_DIR=/home/user/.config/Ultralytics
WORKDIR /home/user/app

RUN git clone --depth 1 https://github.com/RAZORRZR0/bus-safety-ai.git . \
    && pip install --no-cache-dir --user -r requirements.txt \
    && sh scripts/download_models.sh \
    && mkdir -p data \
    && python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# Nano YOLO keeps scans quick on the free 2-vCPU tier.
ENV DEMO_MODE=true REQUIRE_2FA=false YOLO_MODEL=yolo26n.pt \
    DB_URL=sqlite:////home/user/app/data/transport.db PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
