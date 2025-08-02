import asyncio
import logging
import os
import traceback
import json
import time
from typing import List
from functools import partial

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI

from src.document_processor import process_document
from src.redis_client import (
    redis_client,
    check_redis_connection,
    store_embeddings_in_redis,
    search_similar_documents,
)

load_dotenv()

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


@app.get("/api/v1/redis_status")
async def status():
    if check_redis_connection():
        return {"redis_status": "connected"}
    logger.warning("Status check: Redis connection failed")
    return {"redis_status": "disconnected"}


client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1"
)

document_cache = {}
EMBEDDING_BATCH_SIZE = 50


async def run_in_executor(func):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func)


async def generate_embeddings_async(
    chunks: List[str], batch_size: int = EMBEDDING_BATCH_SIZE
) -> List[List[float]]:
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    cleaned_chunks = [c.strip() for c in chunks if c and c.strip()]
    if not cleaned_chunks:
        logger.warning("No valid content to embed after cleaning.")
        return [], 0

    all_embeddings = []
    batch_count = 0

    async def process_batch(batch):
        nonlocal batch_count
        batch_count += 1
        try:
            func = partial(
                client.embeddings.create,
                input=batch,
                model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
                encoding_format="float",
                extra_body={"input_type": "passage", "truncate": "NONE"},
            )
            response = await run_in_executor(func)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding batch failed. Batch contents: {batch}")
            logger.error(f"Error in embedding batch: {str(e)}")
            return []

    tasks = [
        process_batch(cleaned_chunks[i : i + batch_size])
        for i in range(0, len(cleaned_chunks), batch_size)
    ]
    results = await asyncio.gather(*tasks)
    for res in results:
        all_embeddings.extend(res)

    return all_embeddings, batch_count


async def generate_answer_async(question: str, matched_texts: List[str]) -> str:
    logger.info(f"Generating answer for question: {question}")
    context = "\n\n".join(matched_texts)
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Your response should be direct, concise, and detailed. Return only the factual answer, and try to stay as verbatim to the original text as possible, without introductions, conclusions, or additional comments.
"""
    try:

        def get_completion():
            answer = ""
            completion = client.chat.completions.create(
                model="nvidia/llama3-chatqa-1.5-70b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
            return answer

        return await run_in_executor(get_completion)
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "Failed to generate an answer due to an error."


log_entry_counter = 0


async def log_qa_to_file(request_body, answers=None, timing_data=None, metrics=None):
    try:
        global log_entry_counter
        log_entry_counter += 1

        log_data = {
            "entry_id": log_entry_counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "document_url": request_body.get("documents", ""),
        }

        if metrics:
            log_data["metrics"] = metrics

        if answers:
            qa_pairs = []
            for i, (q, a) in enumerate(zip(request_body.get("questions", []), answers)):
                qa_pairs.append({"question_id": i + 1, "question": q, "answer": a})
            log_data["qa_pairs"] = qa_pairs

        if timing_data:
            log_data["performance_metrics"] = timing_data

        with open("requests.log", "a") as log_file:
            log_file.write(json.dumps(log_data, indent=2) + ",\n\n")

    except Exception as e:
        logger.error(f"Error logging to file: {str(e)}")


@app.post("/api/v1/hackrx/run")
async def process_documents(request: Request):
    start_time = time.time()
    timing_data = {}
    metrics = {}

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid or missing JSON in request body"
        )

    if "documents" not in body or "questions" not in body:
        raise HTTPException(
            status_code=400, detail="Request must include 'documents' and 'questions'"
        )

    document_url = body["documents"]
    questions = body["questions"]

    try:
        doc_process_start = time.time()
        if document_url not in document_cache or body.get("force_reprocess", False):
            logger.info(f"Processing document: {document_url}")
            chunks = await process_document(document_url, 1000, 100)
            document_cache[document_url] = {
                "chunks": chunks,
                "timestamp": str(asyncio.get_event_loop().time()),
            }
            timing_data["document_fetching_and_processing"] = round(
                time.time() - doc_process_start, 3
            )
        else:
            logger.info(f"Using cached document for {document_url}")
            chunks = document_cache[document_url]["chunks"]
            timing_data["document_cache_retrieval"] = round(
                time.time() - doc_process_start, 3
            )

        metrics["document_chunks_count"] = len(chunks)
        metrics["avg_chunk_length"] = (
            round(sum(len(chunk) for chunk in chunks) / len(chunks), 1) if chunks else 0
        )

        embedding_start = time.time()
        combined_input = chunks + questions
        all_embeddings, _ = await generate_embeddings_async(combined_input)
        timing_data["embedding_generation"] = round(time.time() - embedding_start, 3)

        chunk_embeddings = all_embeddings[: len(chunks)]
        question_embeddings = all_embeddings[len(chunks) :]

        redis_start = time.time()
        await store_embeddings_in_redis(chunks, chunk_embeddings)
        timing_data["storing_embeddings_in_redis"] = round(time.time() - redis_start, 3)

        async def process_question_answer(question, q_embedding):
            search_start = time.time()
            search_result = await search_similar_documents(q_embedding, k=7)
            search_time = round(time.time() - search_start, 3)

            answer_start = time.time()
            matched_texts = [doc["text"] for doc in search_result]
            answer = await generate_answer_async(question, matched_texts)
            answer_time = round(time.time() - answer_start, 3)

            return answer

        answer_process_start = time.time()
        answer_tasks = [
            process_question_answer(q, q_emb)
            for q, q_emb in zip(questions, question_embeddings)
        ]
        answers = await asyncio.gather(*answer_tasks)
        timing_data["total_answer_processing"] = round(
            time.time() - answer_process_start, 3
        )
        timing_data["total_request_time"] = round(time.time() - start_time, 3)

        await log_qa_to_file(body, answers, timing_data, metrics)

        return {"answers": answers}

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )
