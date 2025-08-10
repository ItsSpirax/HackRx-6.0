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

# Redis connection configuration
redis_host = os.getenv("REDIS_HOST")
redis_port = int(os.getenv("REDIS_PORT"))
redis_db = int(os.getenv("REDIS_DB"))


def get_redis_client():
    """Create synchronous Redis client for basic operations"""
    try:
        client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        client.ping()
        return client
    except:
        return None


def normalize_vector(vec: List[float]) -> bytes:
    """
    Normalize vector to unit length for consistent similarity calculations
    Redis uses Inner Product distance, so normalization enables cosine similarity
    """
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


def generate_embedding_cache_key(doc_hash: str, embedding: List[float]) -> str:
    """Generate unique cache key for search results based on document and query embedding"""
    key_hash = hashlib.sha256(
        np.array(embedding, dtype=np.float32).tobytes()
    ).hexdigest()
    return f"simsearch:{doc_hash}:{key_hash}"


def generate_qa_cache_key(doc_hash: str, question: str) -> str:
    """Generate unique cache key for question-answer pairs"""
    question_hash = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
    return f"qa_cache:{doc_hash}:{question_hash}"


def escape_redis_query(text: str) -> str:
    """Escape special characters in Redis search queries to prevent syntax errors"""
    specials = r'{}[]()|&!@^~":;,.<>?*$%\'\\/+-'
    return "".join(f"\\{c}" if c in specials else c for c in text)


async def store_embeddings_in_redis(
    doc_hash: str, chunks: List[str], embeddings: List[List[float]]
):
    """
    Store document chunks and their embeddings in Redis vector database
    Creates search index if it doesn't exist and loads document vectors
    """
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=False
    )
    await async_client.ping()

    # Remove duplicate chunks to avoid redundant storage
    unique_chunks = {}
    for chunk, emb in zip(chunks, embeddings):
        if chunk not in unique_chunks:
            unique_chunks[chunk] = emb

    # Determine embedding dimensions for index configuration
    dims = len(next(iter(unique_chunks.values()))) if unique_chunks else 2048

    # Define Redis search index schema for vector and text search
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
                    "distance_metric": "IP",  # Inner Product for normalized vectors
                },
            },
        ],
    )

    # Create search index using RedisVL
    index = AsyncSearchIndex(
        schema=schema,
        redis_client=async_client,
        validate_on_load=True,
    )

    if not await index.exists():
        await index.create(overwrite=True)

    # Prepare documents for insertion with normalized embeddings
    documents = [
        {
            "id": f"doc:{doc_hash}:{i}",
            "doc_hash": doc_hash,
            "text": chunk,
            "embedding": normalize_vector(emb),
        }
        for i, (chunk, emb) in enumerate(unique_chunks.items())
    ]

    # Load documents into the search index
    await index.load(documents)

    # Store document metadata for tracking
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
    """
    Perform hybrid search combining vector similarity and text matching
    Uses both KNN vector search and BM25 text search, then merges results
    """
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=True
    )
    await async_client.ping()

    # Check for cached search results
    cache_key = generate_embedding_cache_key(doc_hash, query_embedding)
    cached = await async_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Normalize query vector for consistent similarity calculation
    query_vec = normalize_vector(query_embedding)

    vector_matches = []
    bm25_matches = []

    try:
        # Vector similarity search using KNN
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
        # BM25 text search for keyword matching
        bm25_query_str = f'@doc_hash:{{{doc_hash}}} "{escape_redis_query(query_text)}"'
        bm25_query = (
            Query(bm25_query_str).return_fields("text").paging(0, k * 2).dialect(2)
        )
        bm25_res = await async_client.ft("document_index").search(bm25_query)
        bm25_matches = [{"text": doc.text} for doc in bm25_res.docs]
    except Exception as e:
        print(f"BM25 search error: {e}")

    # Merge results from both search methods, removing duplicates
    seen = set()
    merged_results = []
    for match in vector_matches + bm25_matches:
        if match["text"] not in seen:
            seen.add(match["text"])
            merged_results.append({"text": match["text"]})
        if len(merged_results) == k:
            break

    # Cache results for future queries (1 hour TTL)
    await async_client.set(cache_key, json.dumps(merged_results), ex=3600)
    await async_client.close()

    return merged_results


async def get_cached_answer(doc_hash: str, question: str) -> str:
    """Retrieve cached answer for a specific question-document pair"""
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=True
    )

    try:
        await async_client.ping()
        cache_key = generate_qa_cache_key(doc_hash, question)

        cached_answer = await async_client.get(cache_key)
        return cached_answer if cached_answer else None
    except Exception as e:
        print(f"Error getting cached answer: {e}")
        return None
    finally:
        await async_client.close()


async def cache_answer(doc_hash: str, question: str, answer: str, ttl: int = 86400):
    """Store generated answer in cache with configurable TTL (default: 24 hours)"""
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=True
    )

    try:
        await async_client.ping()
        cache_key = generate_qa_cache_key(doc_hash, question)

        await async_client.set(cache_key, answer, ex=ttl)
    except Exception as e:
        print(f"Error caching answer: {e}")
    finally:
        await async_client.close()


async def clear_qa_cache_for_document(doc_hash: str):
    """Clear all cached Q&A pairs for a specific document when force refresh is used"""
    async_client = redis_async.Redis(
        host=redis_host, port=redis_port, db=redis_db, decode_responses=True
    )

    try:
        await async_client.ping()
        pattern = f"qa_cache:{doc_hash}:*"
        cursor = 0
        deleted_count = 0
        # Use scan to safely iterate through keys matching the pattern
        while True:
            cursor, keys = await async_client.scan(
                cursor=cursor, match=pattern, count=100
            )
            if keys:
                deleted_count += await async_client.delete(*keys)
            if cursor == 0:
                break
        print(f"Cleared {deleted_count} cached Q&A pairs for document {doc_hash}")
    except Exception as e:
        print(f"Error clearing QA cache: {e}")
    finally:
        await async_client.close()
