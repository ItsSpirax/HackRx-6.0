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

invoke_url = os.getenv("RERANKING_INVOKE_URL")

embeddingsClient = OpenAI(
    api_key=os.getenv("NV_EMBEDDINGS_API_KEY"),
    base_url=os.getenv("EMBEDDINGS_BASE_URL"),
)

document_cache = {}
EMBEDDING_BATCH_SIZE = 50
LAST_USED_API_KEY_NV = 0
LAST_USED_API_KEY_GEMINI = 0


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
                model=os.getenv("EMBEDDING_MODEL"),
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

Use citation format [CITE:<source_number>] after every factual statement.

Context:
{context}

Question: {question}

Rules:
- Answer only using the provided context.
- Be direct, concise, and factual.
- If the answer is missing, contradictory, or unclear in context, reply: "I am unable to answer this question based on the provided documents."
- Do not include any introductions, summaries, or markdown.
"""
    try:

        def get_completion():
            global LAST_USED_API_KEY_GEMINI
            if LAST_USED_API_KEY_GEMINI == 0:
                api_key = os.getenv("GEMINI_COMPLETION_API_KEY_0")
            elif LAST_USED_API_KEY_GEMINI == 1:
                api_key = os.getenv("GEMINI_COMPLETION_API_KEY_1")
            elif LAST_USED_API_KEY_GEMINI == 2:
                api_key = os.getenv("GEMINI_COMPLETION_API_KEY_2")
            else:
                api_key = os.getenv("GEMINI_COMPLETION_API_KEY_3")

            LAST_USED_API_KEY_GEMINI = (LAST_USED_API_KEY_GEMINI + 1) % 4

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                os.getenv("COMPLETION_MODEL"),
                system_instruction="You are a professional research assistant that only provides answers based on the documents provided. Never invent information.",
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS")),
                    temperature=float(os.getenv("TEMPERATURE")),
                    top_p=float(os.getenv("TOP_P")),
                    top_k=int(os.getenv("TOP_K")),
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
        logger.info(f"Processing document: {document_url}")
        results = await process_document(
            document_url, force_refresh=body.get("force_refresh", False)
        )
        doc_hash = results["doc_hash"]
        metrics["document_chunks_count"] = results["no_of_chunks"]
        metrics["average_tokens"] = results["average_tokens"]

        if results["status"] != "cached":
            embedding_start = time.time()
            combined_input = results["chunks"] + questions
            all_embeddings, _ = await generate_embeddings_async(combined_input)
            chunk_embeddings = all_embeddings[: len(results["chunks"])]
            question_embeddings = all_embeddings[len(results["chunks"]) :]

            timing_data["embedding_generation"] = round(
                time.time() - embedding_start, 3
            )
            redis_start = time.time()
            await store_embeddings_in_redis(
                doc_hash, results["chunks"], chunk_embeddings
            )
            timing_data["storing_embeddings_in_redis"] = round(
                time.time() - redis_start, 3
            )
        else:
            logger.info(
                "Using cached embeddings for document. Generating embeddings for questions only."
            )
            embedding_start = time.time()
            question_embeddings, _ = await generate_embeddings_async(questions)
            timing_data["embedding_generation"] = round(
                time.time() - embedding_start, 3
            )
            timing_data["storing_embeddings_in_redis"] = 0

        async def process_question_answer(question, q_embedding):
            search_result = await search_similar_documents(
                q_embedding, doc_hash, k=int(os.getenv("SEARCH_RESULTS_COUNT"))
            )

            passages = [{"text": doc["text"]} for doc in search_result]

            payload = {
                "model": os.getenv("RERANKING_MODEL"),
                "query": {"text": question},
                "passages": passages,
            }

            def rerank_documents():
                try:
                    global LAST_USED_API_KEY_NV
                    session = requests.Session()
                    api_key = (
                        os.getenv("NV_RANKING_API_KEY_0")
                        if LAST_USED_API_KEY_NV == 0
                        else os.getenv("NV_RANKING_API_KEY_1")
                    )
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                    }
                    LAST_USED_API_KEY_NV = 1 - LAST_USED_API_KEY_NV
                    response = session.post(invoke_url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    logger.error(f"Error in reranking: {str(e)}")
                    return {
                        "rankings": [
                            {"index": i, "logit": 0}
                            for i in range(
                                min(int(os.getenv("TOP_RESULTS_COUNT")), len(passages))
                            )
                        ]
                    }

            rerank_result = await run_in_executor(rerank_documents)

            rankings = rerank_result.get("rankings", [])
            sorted_rankings = sorted(
                rankings, key=lambda x: x.get("logit", 0), reverse=True
            )
            top_indices = [
                item["index"]
                for item in sorted_rankings[: int(os.getenv("TOP_RESULTS_COUNT"))]
            ]
            matched_texts = [
                passages[idx]["text"] for idx in top_indices if idx < len(passages)
            ]

            if len(matched_texts) < int(os.getenv("TOP_RESULTS_COUNT")):
                remaining = int(os.getenv("TOP_RESULTS_COUNT")) - len(matched_texts)
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
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        logger.info(f"Timing Data: {json.dumps(timing_data, indent=2)}")
        logger.info(
            f"Processed {len(questions)} questions in {timing_data['total_request_time']} seconds"
        )

        return {"answers": answers}

    except Exception as e:
        if "Unsupported document type" in str(e):
            logger.error(f"Unsupported document type: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Unsupported document type")
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )
