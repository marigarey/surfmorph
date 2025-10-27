FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --extra-index-url https://pypi.org/simple
RUN pip install -r /app/requirements.txt


COPY . /app
ENV PYTHONPATH=/app/src


ENTRYPOINT ["python", "src/train.py"]