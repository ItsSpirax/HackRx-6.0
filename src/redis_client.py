import redis
import os
import logging

redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_db = int(os.getenv("REDIS_DB", 0))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("redis_client")
logger.info(f"Redis configuration: host={redis_host}, port={redis_port}, db={redis_db}")


def get_redis_client():
    try:
        logger.info(f"Attempting to connect to Redis at {redis_host}:{redis_port}...")
        client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        client.ping()
        logger.info("Redis connection successful")
        return client
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection failed: {e}")
        return None

redis_client = get_redis_client()

def check_redis_connection():
    if redis_client:
        try:
            if redis_client.ping():
                return True
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"Redis connection check failed: {e}")
    return False
