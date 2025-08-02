import email
import io
import re
from typing import List
from urllib.parse import urlparse

import docx
import requests
from PyPDF2 import PdfReader
from transformers import AutoTokenizer

llama3_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")


def identify_document_type(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf") or "pdf" in url:
        return "pdf"
    if path.endswith(".docx") or "docx" in url:
        return "docx"
    if path.endswith(".eml") or "email" in url or "eml" in url:
        return "email"
    return "pdf"


def download_document(url: str) -> bytes:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to download document. Status code: {response.status_code}"
        )
    return response.content


def extract_text_from_pdf(content: bytes) -> str:
    return " ".join(
        page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages
    )


def extract_text_from_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    return " ".join(p.text for p in doc.paragraphs if p.text)


def extract_text_from_email(content: bytes) -> str:
    msg = email.message_from_bytes(content)
    if msg.is_multipart():
        return " ".join(
            part.get_payload(decode=True).decode("utf-8", "ignore")
            for part in msg.get_payload()
            if part.get_content_type() == "text/plain"
        )
    return msg.get_payload(decode=True).decode("utf-8", "ignore")


def get_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def token_length(text: str) -> int:
    return len(llama3_tokenizer.encode(text, add_special_tokens=False))


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = get_sentences(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_length(sentence)
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            overlap_chunk = []
            overlap_count = 0
            for s in reversed(current_chunk):
                tok_len = token_length(s)
                if overlap_count + tok_len <= overlap_tokens:
                    overlap_chunk.insert(0, s)
                    overlap_count += tok_len
                else:
                    break
            current_chunk = overlap_chunk + [sentence]
            current_tokens = token_length(" ".join(current_chunk))

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


async def process_document(
    url: str, chunk_size: int = 512, chunk_overlap: int = 100
) -> List[str]:
    doc_type = identify_document_type(url)
    content = download_document(url)

    if doc_type == "pdf":
        text = extract_text_from_pdf(content)
    elif doc_type == "docx":
        text = extract_text_from_docx(content)
    elif doc_type == "email":
        text = extract_text_from_email(content)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    return chunk_text(text, chunk_size, chunk_overlap)
