import asyncio
import logging
import os
import traceback
import json
import time
import re
import requests
from typing import List
from functools import partial

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
import google.generativeai as genai
from google.generativeai import types
from openai import OpenAI

from src.document_processor import process_document
from src.redis_client import (
    redis_client,
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

invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nemoretriever-500m-rerank-v2/reranking"

genai.configure(api_key=os.getenv("GEMINI_COMPLETION_API_KEY"))

embeddingsClient = OpenAI(
    api_key=os.getenv("NV_EMBEDDINGS_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)

document_cache = {}
EMBEDDING_BATCH_SIZE = 50
LAST_USED_API_KEY = 0


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
                embeddingsClient.embeddings.create,
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
    context = "\n\n".join(
        [f"Context {i+1}: {text}" for i, text in enumerate(matched_texts)]
    )
    prompt = f"""
You are a helpful assistant who provides direct and concise answers based on the provided context.

Context:
{context}

Question: {question}

Rules:
- Answer the question based *only* on the information in the provided context.
- Your response must be direct, concise, and factual.
- For every piece of information you provide, append a citation `[CITE:<source_number>]` where `<source_number>` corresponds to the context number (e.g., [CITE:1], [CITE:2]).
- If the answer is not found in the context, or if the context provides contradictory or insufficient information, respond with "I am unable to answer this question based on the provided documents."
- Do not use any introductory phrases, conclusions, or markdown.
- Keep the response as short as possible while including all relevant details.
"""
    try:

        def get_completion():
            model = genai.GenerativeModel(
                "gemini-2.0-flash-lite",
                system_instruction="You are a professional research assistant that only provides answers based on the documents provided. Never invent information.",
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=512,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=5,
                ),
            )
            return response.text if hasattr(response, "text") else str(response)

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
            search_result = await search_similar_documents(q_embedding, k=20)

            passages = [{"text": doc["text"]} for doc in search_result]

            payload = {
                "model": "nvidia/llama-3.2-nemoretriever-500m-rerank-v2",
                "query": {"text": question},
                "passages": passages,
            }

            def rerank_documents():
                try:
                    global LAST_USED_API_KEY
                    session = requests.Session()
                    api_key = (
                        os.getenv("NV_RANKING_API_KEY_0")
                        if LAST_USED_API_KEY == 0
                        else os.getenv("NV_RANKING_API_KEY_1")
                    )
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                    }
                    LAST_USED_API_KEY = 1 - LAST_USED_API_KEY
                    response = session.post(invoke_url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    logger.error(f"Error in reranking: {str(e)}")
                    return {
                        "rankings": [
                            {"index": i, "logit": 0}
                            for i in range(min(10, len(passages)))
                        ]
                    }

            rerank_result = await run_in_executor(rerank_documents)

            rankings = rerank_result.get("rankings", [])
            sorted_rankings = sorted(
                rankings, key=lambda x: x.get("logit", 0), reverse=True
            )
            top_indices = [item["index"] for item in sorted_rankings[:10]]
            matched_texts = [
                passages[idx]["text"] for idx in top_indices if idx < len(passages)
            ]

            if len(matched_texts) < 10:
                remaining = 10 - len(matched_texts)
                original_texts = [p["text"] for p in passages]
                for text in original_texts:
                    if text not in matched_texts and remaining > 0:
                        matched_texts.append(text)
                        remaining -= 1
                    if remaining == 0:
                        break

            answer = await generate_answer_async(question, matched_texts)
            answer = answer.replace("\n", " ").strip()
            cleaned_answer = re.sub(r" \[CITE:.*?\]", "", answer)
            return cleaned_answer

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
