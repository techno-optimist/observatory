FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install dependencies (no TRL - using standard Trainer)
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate \
    peft>=0.10.0 \
    datasets \
    bitsandbytes \
    safetensors \
    sentencepiece \
    huggingface_hub \
    tensorboard

# Set working directory
WORKDIR /app

# Copy training script
COPY train.py .

# Set HF cache directory
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

# Run training on start
CMD ["python", "train.py"]
