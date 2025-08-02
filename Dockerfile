FROM python:3.12-slim-bookworm AS builder
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

FROM python:3.12-slim-bookworm AS production
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY . .
RUN adduser --system --group hackrx
ENV HOME=/home/hackrx \
    PATH=/home/hackrx/.local/bin:$PATH

RUN mkdir -p /home/hackrx/.cache && \
    chown -R hackrx:hackrx /home/hackrx/.cache && \
    chmod -R 755 /home/hackrx/.cache

USER hackrx
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]