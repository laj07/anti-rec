FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "src/main.py"]
CMD ["--config", "experiments/movies/config_1m.yaml", "--mode", "train"]
