import email
import io
import re
import hashlib
from pathlib import Path
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


def extract_structured_docx(content: bytes) -> List[str]:
    doc = docx.Document(io.BytesIO(content))
    chunks = []
    current_section = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            if current_section:
                chunks.append("\n".join(current_section))
                current_section = []
            current_section.append(f"## {text}")
        elif para.style.name.startswith("List"):
            current_section.append(f"- {text}")
        else:
            current_section.append(text)

    if current_section:
        chunks.append("\n".join(current_section))

    return chunks


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


def smart_chunk_sections(
    sections: List[str], max_tokens: int, overlap_tokens: int
) -> List[str]:
    final_chunks = []
    for section in sections:
        tokens = llama3_tokenizer.encode(section, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            final_chunks.append(section)
        else:
            sentences = get_sentences(section)
            chunk = []
            chunk_len = 0
            for sent in sentences:
                sent_tokens = token_length(sent)
                if chunk_len + sent_tokens <= max_tokens:
                    chunk.append(sent)
                    chunk_len += sent_tokens
                else:
                    final_chunks.append(" ".join(chunk))
                    chunk = [sent]
                    chunk_len = sent_tokens
            if chunk:
                final_chunks.append(" ".join(chunk))
    return final_chunks


async def process_document(
    url: str, chunk_size: int = 512, chunk_overlap: int = 100
) -> List[str]:
    doc_type = identify_document_type(url)

    document_data_dir = Path("document_data")
    document_data_dir.mkdir(exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()
    filename = f"{url_hash}.{doc_type}"
    file_path = document_data_dir / filename

    if file_path.exists():
        with open(file_path, "rb") as f:
            content = f.read()
    else:
        content = download_document(url)
        with open(file_path, "wb") as f:
            f.write(content)

    if doc_type == "pdf":
        text = extract_text_from_pdf(content)
        return smart_chunk_sections([text], chunk_size, chunk_overlap)
    elif doc_type == "docx":
        structured_sections = extract_structured_docx(content)
        return smart_chunk_sections(structured_sections, chunk_size, chunk_overlap)
    elif doc_type == "email":
        text = extract_text_from_email(content)
        return smart_chunk_sections([text], chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")
