FROM python:3.12-slim-bookworm AS builder
WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r /app/requirements.txt


FROM python:3.12-slim-bookworm AS production
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libreoffice-impress \
    libreoffice-core \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --system --group deepintel

RUN chown -R deepintel:deepintel /app

ENV HOME=/home/deepintel \
    PATH=/home/deepintel/.local/bin:$PATH

ENV PYTHONPATH=/app:$PYTHONPATH

RUN mkdir -p /home/deepintel/.cache && \
    chown -R deepintel:deepintel /home/deepintel/.cache && \
    chmod -R 755 /home/deepintel/.cache

RUN mkdir -p /tmp/libreprofile && \
    chown -R deepintel:deepintel /tmp/libreprofile && \
    chmod -R 755 /tmp/libreprofile

RUN mkdir -p /app/document_data && \
    chown -R deepintel:deepintel /app/document_data && \
    chmod -R 755 /app/document_data

RUN mkdir -p /app/uploaded_documents && \
    chown -R deepintel:deepintel /app/uploaded_documents && \
    chmod -R 755 /app/uploaded_documents
    
RUN touch /app/requests.log && \
    chown deepintel:deepintel /app/requests.log && \
    chmod 644 /app/requests.log

USER deepintel
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]