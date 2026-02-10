# CarSee Processor - run ONNX models in the cloud (Render.com)
FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY model_config.py .
COPY inference.py .
COPY main.py .

# Models are downloaded at runtime (lifespan) from GitHub release

EXPOSE 8000
ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
