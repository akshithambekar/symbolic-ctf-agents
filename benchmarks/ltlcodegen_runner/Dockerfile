FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/runner
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY translators /opt/runner/translators
COPY worker.py /opt/runner/worker.py
USER 65534:65534
WORKDIR /tmp
CMD ["python", "/opt/runner/worker.py"]
