# pgvector-ann/backend/src/vectorizer.py
import os
import pandas as pd
from pypdf import PdfReader
from openai import AzureOpenAI, OpenAI
import logging
from config import *
from langchain_text_splitters import CharacterTextSplitter
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.info("Initializing vectorizer to read from local PDF folder")

if ENABLE_OPENAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("Using OpenAI API for embeddings")
else:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    logger.info("Using Azure OpenAI API for embeddings")

def get_pdf_files_from_local():
    pdf_files = [f for f in os.listdir(PDF_INPUT_DIR) if f.endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files in {PDF_INPUT_DIR}")
    return pdf_files

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            return [{"page_content": page.extract_text(), "metadata": {"page": i + 1}} for i, page in enumerate(pdf.pages)]
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return []

def create_embedding(text):
    if ENABLE_OPENAI:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
    else:
        response = client.embeddings.create(
            input=text,
            model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
        )
    return response

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=SEPARATOR
    )
    chunks = text_splitter.split_text(text)
    return chunks if chunks else [text]

def process_pdf(file_name):
    file_path = os.path.join(PDF_INPUT_DIR, file_name)
    pages = extract_text_from_pdf(file_path)
    if not pages:
        logger.warning(f"No text extracted from PDF file: {file_name}")
        return None

    processed_data = []
    total_chunks = 0
    for page in pages:
        page_text = page["page_content"]
        page_num = page["metadata"]["page"]
        chunks = split_text_into_chunks(page_text)

        if not chunks and page_text:
            chunks = [page_text]

        for chunk in chunks:
            if chunk.strip():  # Only process non-empty chunks
                response = create_embedding(chunk)
                current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                total_chunks += 1
                processed_data.append({
                    'file_name': file_name,
                    'document_page': str(page_num),
                    'chunk_no': total_chunks,
                    'chunk_text': chunk,
                    'model': response.model,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'created_date_time': current_time,
                    'chunk_vector': response.data[0].embedding
                })

    logger.info(f"Processed {file_name}: {len(pages)} pages, {total_chunks} chunks")
    return pd.DataFrame(processed_data)

def process_pdf_files():
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    for file_name in get_pdf_files_from_local():
        processed_data = process_pdf(file_name)
        if processed_data is not None and not processed_data.empty:
            csv_file_name = f'{os.path.splitext(file_name)[0]}.csv'
            output_file = os.path.join(CSV_OUTPUT_DIR, csv_file_name)
            processed_data.to_csv(output_file, index=False)
            logger.info(f"Processed data for {file_name} saved to {output_file}")
        else:
            logger.warning(f"No data processed for {file_name}")

if __name__ == "__main__":
    process_pdf_files()
