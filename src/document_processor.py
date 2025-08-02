import requests
import io
import re
from urllib.parse import urlparse
from typing import List
from PyPDF2 import PdfReader
import docx
import email


async def process_document(
    url: str, chunk_size: int = 1000, chunk_overlap: int = 200
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
        page.extract_text()
        for page in PdfReader(io.BytesIO(content)).pages
        if page.extract_text()
    )


def extract_text_from_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    return " ".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)


def extract_text_from_email(content: bytes) -> str:
    msg = email.message_from_bytes(content)
    if msg.is_multipart():
        return " ".join(
            part.get_payload(decode=True).decode("utf-8", "ignore")
            for part in msg.get_payload()
            if part.get_content_type() == "text/plain"
        )
    return msg.get_payload(decode=True).decode("utf-8", "ignore")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if p.strip()
    ]
    chunks = []

    def get_sentences(t):
        return [
            s.strip()
            for s in re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", t)
            if s.strip()
        ]

    current = ""
    for p in paragraphs:
        if len(current) + len(p) > chunk_size and current:
            chunks.append(current.strip())
            overlap = ""
            for s in reversed(get_sentences(current)):
                if len(overlap) + len(s) <= chunk_overlap:
                    overlap = s + " " + overlap
                else:
                    break
            current = overlap + p
        else:
            current += (" " if current else "") + p

    if current:
        chunks.append(current.strip())

    result = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            result.append(chunk)
        else:
            current = ""
            for s in get_sentences(chunk):
                if len(current) + len(s) <= chunk_size:
                    current += (" " if current else "") + s
                else:
                    result.append(current)
                    overlap = ""
                    for os in reversed(get_sentences(current)):
                        if len(overlap) + len(os) <= chunk_overlap:
                            overlap = os + " " + overlap
                        else:
                            break
                    current = overlap + s
            if current:
                result.append(current)
    return result
