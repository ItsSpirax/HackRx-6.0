# HackRX-6.0: Document-Powered AI Assistant

## Project Overview

Team Longg Shott's Solution processes documents and provides answers to user questions. It builds a knowledge base from various document types (PDFs, DOCX, Emails). The system uses **Google's `gemini-embedding-001`** for searching and **Google Gemini** to create answers.

### Key Features:
* **Multi-Document Support:** Processes text from PDFs, DOCX files, and email formats.
* **Persistent Storage:** Stores uploaded documents using Docker volumes.
* **Scalable Ingestion:** Documents are uploaded via an API and processed in the background (extracting text, chunking, and creating embeddings).
* **Vector Database (Redis Stack):** Uses Redis Stack's RediSearch for vector storage and semantic search.
* **Semantic Search:** Finds relevant document parts based on meaning, not just keywords.
* **Retrieval Augmented Generation (RAG):** Uses retrieved document content with Google Gemini to answer questions.
* **Dockerized Deployment:** Easy to set up with Docker Compose.

## Technologies Used

* **Python 3.12+**
* **FastAPI:** Builds the API.
* **Uvicorn:** Runs the FastAPI application.
* **Redis Stack:** In-memory data store with RediSearch for vector search.
* **Google `gemini-embedding-001`**: Used for generating document embeddings.
* **Google Gemini**: The Large Language Model used for generating answers.
* **LangChain:** Helps build the RAG pipeline (loading, splitting documents, talking to Redis and Gemini).
* **Docker & Docker Compose:** For running services in containers.
* **Libraries for Document Parsing:** `pypdf`, `python-docx`, `mail-parser`.