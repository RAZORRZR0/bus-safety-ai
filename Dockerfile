FROM python:3.12-slim

# OpenCV runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only torch keeps the image a few GB smaller
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN sh scripts/download_models.sh && mkdir -p data

ENV DB_URL=sqlite:////app/data/transport.db
EXPOSE 8000
# PORT lets hosts like Hugging Face Spaces (7860) pick the port.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
