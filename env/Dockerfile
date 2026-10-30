FROM python:3.11-slim
WORKDIR /app
COPY env/requirements.lock /app/env/requirements.lock
RUN pip install --no-cache-dir -r /app/env/requirements.lock
COPY . /app
CMD ["python", "-m", "src.pipelines.all"]

