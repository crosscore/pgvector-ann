# pgvector-ann/backend/src/vectorizer.py
import os
import pandas as pd
from pypdf import PdfReader
from openai import AzureOpenAI, OpenAI
import logging
from config import *
from langchain_text_splitters import CharacterTextSplitter
from datetime import datetime, timezone

logging.basicConfig(filename="/app/data/log/vectorizer.log", level=logging.INFO, format='%(asctime)s - %(message)s')
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
    pdf_files = []
    for root, _, files in os.walk(PDF_INPUT_DIR):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
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

def get_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f}MB"

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
    all_data = []
    for file_path in get_pdf_files_from_local():
        processed_data = process_pdf(file_path)
        if processed_data is not None and not processed_data.empty:
            relative_path = os.path.relpath(file_path, PDF_INPUT_DIR)
            output_dir = os.path.join(CSV_OUTPUT_DIR, os.path.dirname(relative_path))
            os.makedirs(output_dir, exist_ok=True)

            csv_file_name = f'{os.path.splitext(os.path.basename(file_path))[0]}.csv'
            output_file = os.path.join(output_dir, csv_file_name)

            processed_data.to_csv(output_file, index=False)
            logger.info(f"CSV output completed for {csv_file_name}")

            all_data.append(processed_data)
        else:
            logger.warning(f"No data processed for {file_path}")

    # Combine all processed data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

        os.makedirs(os.path.dirname("/app/data/csv/all/all.csv"), exist_ok=True)
        combined_data.to_csv("/app/data/csv/all/all.csv", index=False)
        logger.info("CSV output completed for all.csv")
    else:
        logger.warning("No data processed. all.csv was not created.")

if __name__ == "__main__":
    process_pdf_files()
    