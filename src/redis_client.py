import asyncio
import json
import os
from typing import List

import numpy as np
import redis
import redis.asyncio as redis_async
from dotenv import load_dotenv
from redis.commands.search.query import Query
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema


load_dotenv()


redis_host = os.getenv("REDIS_HOST")
redis_port = int(os.getenv("REDIS_PORT"))
redis_db = int(os.getenv("REDIS_DB"))


def get_redis_client():
    try:
        client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        client.ping()
        return client
    except:
        return None


redis_client = get_redis_client()


def check_redis_connection():
    if redis_client:
        try:
            return redis_client.ping()
        except:
            pass
    return False


def normalize_vector(vec: List[float]) -> bytes:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


async def store_embeddings_in_redis(chunks: List[str], embeddings: List[List[float]]):
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=False
    )
    await async_client.ping()

    unique_chunks = {}
    for chunk, emb in zip(chunks, embeddings):
        if chunk not in unique_chunks:
            unique_chunks[chunk] = emb

    dims = len(next(iter(unique_chunks.values()))) if unique_chunks else 2048

    schema = IndexSchema(
        index={"name": "document_index", "prefix": "doc:"},
        fields=[
            {"name": "text", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": dims,
                    "distance_metric": "COSINE",
                },
            },
        ],
    )

    index = AsyncSearchIndex(
        schema=schema,
        redis_client=async_client,
        validate_on_load=True,
    )

    if await index.exists():
        await index.delete(drop=True)

    await index.create(overwrite=True)

    documents = [
        {
            "id": f"doc:{i}",
            "text": chunk,
            "embedding": normalize_vector(emb),
        }
        for i, (chunk, emb) in enumerate(unique_chunks.items())
    ]

    await index.load(documents)

    await async_client.set(
        "document:metadata",
        json.dumps(
            {
                "num_chunks": len(unique_chunks),
                "timestamp": str(asyncio.get_event_loop().time()),
            }
        ),
    )

    await async_client.close()


async def search_similar_documents(query_embedding: List[float], k: int = 10):
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=False
    )
    await async_client.ping()

    query_vec = normalize_vector(query_embedding)

    base_query = f"*=>[KNN {k * 2} @embedding $vec_param]"

    q = (
        Query(base_query)
        .return_fields("text")
        .sort_by("__embedding_score")
        .paging(0, k * 2)
        .dialect(2)
    )

    try:
        res = await async_client.ft("document_index").search(
            q, query_params={"vec_param": query_vec}
        )
    except Exception as e:
        await async_client.close()
        raise Exception(f"Error during Redis search: {e}")

    seen = set()
    unique_matches = []
    for doc in res.docs:
        text = doc.text
        if text not in seen:
            seen.add(text)
            unique_matches.append({"text": text})
        if len(unique_matches) == k:
            break

    await async_client.close()
    return unique_matches
