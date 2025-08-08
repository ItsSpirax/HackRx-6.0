# HackRX-6.0: Document-Powered AI Assistant

## Overview

Team Longg Shott's service takes a document URL and a list of questions, extracts and chunks the document, embeds the chunks, retrieves relevant passages from Redis, re-ranks them, and generates concise answers.

Pipeline at a glance:
- Detect document type and download it.
- Extract and clean text (PDF, DOCX, Email, Excel, CSV, HTML, Images, PPT/PPTX via OCR).
- Tokenize and chunk text (configurable size/overlap).
- Generate embeddings for chunks and questions (NVIDIA embeddings API via OpenAI SDK).
- Store chunk vectors in Redis Stack (RediSearch) and search (KNN + BM25 hybrid).
- Re-rank candidates with NVIDIA Rerank API.
- Compose answers with Google Gemini models.

## API

POST /api/v1/hackrx/run
Body:
{
	"documents": "<public URL to the document>",
	"questions": ["question 1", "question 2"],
	"force_refresh": false
}

Response:
{
	"answers": ["answer for q1", "answer for q2"]
}

Notes:
- documents must be a reachable URL. Content type and/or extension are used to detect format.
- force_refresh=true reprocesses the document even if cached metadata exists in Redis.
- Answers are concise; internal citation markers are removed before returning.

## Supported document types

- PDF (PyMuPDF)
- DOCX (python-docx)
- EML/MSG (email parsing)
- XLSX/XLS (openpyxl)
- CSV (csv)
- HTML (BeautifulSoup + lxml)
- JPG/PNG via OCR (Mistral OCR)
- PPT/PPTX via LibreOffice -> PDF -> images -> OCR (Mistral OCR)

## Configuration (.env)

Copy .env.example to .env and fill values. Key variables used in code:

- Redis
	- REDIS_HOST (default in compose: redis)
	- REDIS_PORT (default: 6379)
	- REDIS_DB (default: 0)

- Tokenization / chunking
	- TOKENIZER_MODEL (Hugging Face model id)
	- HF_API_KEY (required to download tokenizer if gated)
	- DOCUMENT_CHUNK_SIZE (int, tokens)
	- DOCUMENT_CHUNK_OVERLAP (int, tokens)

- Embeddings (NVIDIA via OpenAI SDK)
	- NV_EMBEDDINGS_API_KEY
	- EMBEDDINGS_BASE_URL (e.g., https://integrate.api.nvidia.com/v1)
	- EMBEDDING_MODEL (e.g., nvidia/nv-embedqa-mistral-7b-v2)
	- EMBEDDING_BATCH_SIZE (int)

- Re-ranking (NVIDIA)
	- RERANKING_INVOKE_URL (e.g., https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking)
	- RERANKING_MODEL (e.g., nv-rerank-qa-mistral-4b:1)
	- NV_RANKING_API_KEY_0, NV_RANKING_API_KEY_1, NV_RANKING_API_KEY_2, NV_RANKING_API_KEY_3 (one or more; rotated per request)

- Completion (Google Gemini)
	- COMPLETION_MODEL (e.g., gemini-2.5-flash-lite)
	- GEMINI_COMPLETION_API_KEY and/or GEMINI_COMPLETION_API_KEY_1/_2/_3 (set at least one; code rotates among those exact names)
	- Optional alternate model if the strict prompt refuses:
		- ALT_COMPLETION_MODEL
		- ALT_MAX_OUTPUT_TOKENS, ALT_TEMPERATURE, ALT_TOP_P, ALT_TOP_K

- Generation controls
	- MAX_OUTPUT_TOKENS, TEMPERATURE, TOP_P, TOP_K
	- SEARCH_RESULTS_COUNT (initial retrieve count)
	- TOP_RESULTS_COUNT (top passages after re-ranking)

- OCR (Images/PPT/PPTX)
	- MISTRAL_API_KEY

Tip: In .env, do not wrap values in quotes. Use KEY=value, not KEY="value".

## Run with Docker Compose

Prerequisites:
- Docker and Docker Compose

Steps (PowerShell):
```powershell
Copy-Item .env.example .env -Force
# Edit .env to add your API keys
docker compose up --build -d
```

The API will be on http://localhost:8000

Quick test:
```powershell
Invoke-RestMethod `
	-Uri http://localhost:8000/api/v1/hackrx/run `
	-Method POST `
	-ContentType 'application/json' `
	-Body (@{
		documents = 'https://example.com/sample.pdf'
		questions = @('What is the main objective?', 'List the key steps.')
		force_refresh = $false
	} | ConvertTo-Json)
```

## Local development (without Docker)

Prerequisites:
- Python 3.12
- Redis Stack running locally (docker run -p 6379:6379 redis/redis-stack-server:latest)
 - For PPT/PPTX OCR and image-to-PDF: install LibreOffice and Poppler (Windows builds available online)

Setup and run:
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
$env:REDIS_HOST = 'localhost'
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## How it works

- Document processing (src/document_processor.py)
	- identify_document_type() uses headers/URL to infer type.
	- download_document() fetches the file.
	- parse_document_content() extracts text per type (including OCR via Mistral for images/PPT/PPTX).
	- chunk_sections() splits text using Hugging Face tokenizer + LangChain splitter.
	- Redis stores per-document metadata; chunk texts are embedded and indexed.

- Embeddings & search (src/redis_client.py)
	- Vectors normalized to unit length; stored with RedisVL index (IP distance).
	- Hybrid retrieval: vector KNN + BM25 on text; results merged and cached per-query.

- Orchestration (src/main.py)
	- POST /api/v1/hackrx/run accepts the URL and questions.
	- Embeds chunks (if not cached) and questions.
	- Retrieves and re-ranks passages, builds a guarded prompt, and calls Gemini.
	- Writes a structured entry to requests.log with timing/metrics.

## Troubleshooting

- 401/403 from NVIDIA or Google: verify API keys in .env and that models are enabled for your account.
- Redis index missing: the service creates it automatically; ensure you are running Redis Stack (not vanilla redis).
- OCR for PPT/PPTX fails in Docker: container installs LibreOffice and poppler-utils; ensure the URL is reachable and documents are not password-protected.
- Tokenizer download issues: set HF_API_KEY and ensure outbound network is allowed.
- Large documents: tune DOCUMENT_CHUNK_SIZE and DOCUMENT_CHUNK_OVERLAP; EMBEDDING_BATCH_SIZE controls throughput.

## Notes

- The compose file mounts only the .env. Redis data is persisted via the redis_data volume.