from fastapi import FastAPI
from src.redis_client import redis_client, check_redis_connection
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("main")
app = FastAPI()
logger.info("Starting Longg Shott API service")


@app.get("/")
async def root():
    return {"status": "API is running", "code": 200}


@app.get("/redis_status")
async def status():
    if check_redis_connection():
        return {"redis_status": "connected"}
    logger.warning("Status check: Redis connection failed")
    return {"redis_status": "disconnected"}
