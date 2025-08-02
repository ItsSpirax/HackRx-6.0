import asyncio
import logging
import os
import traceback
from typing import List

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


async def generate_embeddings(
    chunks: List[str],
) -> List[List[float]]:
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    all_embeddings = []

    try:
        logger.info(
            f"Processing {len(chunks)} chunks with model nvidia/llama-3.2-nemoretriever-300m-embed-v1"
        )
        response = client.embeddings.create(
            input=chunks,
            model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "NONE"},
        )
        all_embeddings.extend([item.embedding for item in response.data])
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e) or "Authorization failed" in str(e):
            logger.error(f"API authorization failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to connect to embedding API: Authorization failed",
            )
        elif "429" in str(e) or "rate limit" in str(e).lower():
            logger.warning(f"Rate limit error. Waiting for 20 seconds...")
            await asyncio.sleep(20)
        else:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
    return all_embeddings


async def generate_answer(question: str, matched_texts: List[str]) -> str:
    logger.info(f"Generating answer for question: {question}")

    context = "\n\n".join(matched_texts)

    prompt = f"""Use the following context to answer the question.
    
Context:
{context}

Question: {question}

Your response should be direct and concise and detailed. Return only the factual answer without introductions, conclusions or additional comments.
"""

    try:
        answer_text = ""
        completion = client.chat.completions.create(
            model="nvidia/llama3-chatqa-1.5-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True,
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                answer_text += chunk.choices[0].delta.content

        logger.info(f"Successfully generated answer of length {len(answer_text)}")
        return answer_text

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "Failed to generate an answer due to an error."


@app.post("/api/v1/hackrx/run")
async def process_documents(request: Request):
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
        chunks = await process_document(document_url)
        chunk_embeddings = await generate_embeddings(chunks)
        await store_embeddings_in_redis(chunks, chunk_embeddings)

        questions_embeddings = await generate_embeddings(questions)

        search_tasks = []
        for q_embedding in questions_embeddings:
            search_tasks.append(search_similar_documents(q_embedding, k=15))

        search_results = await asyncio.gather(*search_tasks)

        answer_tasks = []
        for question, result in zip(questions, search_results):
            matched_texts = [doc["text"] for doc in result]
            answer_tasks.append(generate_answer(question, matched_texts))

        answers = await asyncio.gather(*answer_tasks)

        return {
            "answers": answers,
        }

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )
