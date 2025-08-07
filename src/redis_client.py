import asyncio
import json
import os
import hashlib
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


def normalize_vector(vec: List[float]) -> bytes:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


def generate_embedding_cache_key(doc_hash: str, embedding: List[float]) -> str:
    key_hash = hashlib.sha256(
        np.array(embedding, dtype=np.float32).tobytes()
    ).hexdigest()
    return f"simsearch:{doc_hash}:{key_hash}"


def escape_redis_query(text: str) -> str:
    specials = r'{}[]()|&!@^~":;,.<>?*$%\'\\/+-'
    return "".join(f"\\{c}" if c in specials else c for c in text)


async def store_embeddings_in_redis(
    doc_hash: str, chunks: List[str], embeddings: List[List[float]]
):
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
            {"name": "doc_hash", "type": "tag"},
            {"name": "text", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": dims,
                    "distance_metric": "IP",
                },
            },
        ],
    )

    index = AsyncSearchIndex(
        schema=schema,
        redis_client=async_client,
        validate_on_load=True,
    )

    if not await index.exists():
        await index.create(overwrite=True)

    documents = [
        {
            "id": f"doc:{doc_hash}:{i}",
            "doc_hash": doc_hash,
            "text": chunk,
            "embedding": normalize_vector(emb),
        }
        for i, (chunk, emb) in enumerate(unique_chunks.items())
    ]

    await index.load(documents)

    await async_client.set(
        f"document:metadata:{doc_hash}",
        json.dumps(
            {
                "num_chunks": len(unique_chunks),
                "timestamp": str(asyncio.get_event_loop().time()),
            }
        ),
    )

    await async_client.close()


async def search_similar_documents(
    query_embedding: List[float], query_text: str, doc_hash: str, k: int = 10
):
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=True
    )
    await async_client.ping()

    cache_key = generate_embedding_cache_key(doc_hash, query_embedding)
    cached = await async_client.get(cache_key)
    if cached:
        return json.loads(cached)

    query_vec = normalize_vector(query_embedding)

    vector_matches = []
    bm25_matches = []

    try:
        base_query = f"@doc_hash:{{{doc_hash}}}=>[KNN {k*2} @embedding $vec_param]"
        vector_query = (
            Query(base_query)
            .return_fields("text")
            .sort_by("__embedding_score")
            .paging(0, k * 2)
            .dialect(2)
        )
        vector_res = await async_client.ft("document_index").search(
            vector_query, query_params={"vec_param": query_vec}
        )
        vector_matches = [{"text": doc.text} for doc in vector_res.docs]
    except Exception as e:
        print(f"Vector search error: {e}")

    try:
        bm25_query_str = f'@doc_hash:{{{doc_hash}}} "{escape_redis_query(query_text)}"'
        bm25_query = (
            Query(bm25_query_str).return_fields("text").paging(0, k * 2).dialect(2)
        )
        bm25_res = await async_client.ft("document_index").search(bm25_query)
        bm25_matches = [{"text": doc.text} for doc in bm25_res.docs]
    except Exception as e:
        print(f"BM25 search error: {e}")

    seen = set()
    merged_results = []
    for match in vector_matches + bm25_matches:
        if match["text"] not in seen:
            seen.add(match["text"])
            merged_results.append({"text": match["text"]})
        if len(merged_results) == k:
            break

    await async_client.set(cache_key, json.dumps(merged_results), ex=3600)
    await async_client.close()

    return merged_results
