import os
import pandas as pd
from pypdf import PdfReader
from openai import AzureOpenAI, OpenAI
import logging
from config import *
from langchain_text_splitters import CharacterTextSplitter
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
            return [{"page_content": page.extract_text(), "metadata": {"page": i}} for i, page in enumerate(pdf.pages)]
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

def process_pdf_and_create_dataframe(file_name):
    file_path = os.path.join(PDF_INPUT_DIR, file_name)
    pages = extract_text_from_pdf(file_path)
    if not pages:
        logger.warning(f"No text extracted from PDF file: {file_name}")
        return pd.DataFrame()

    data = []
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
                jst = ZoneInfo("Asia/Tokyo")
                current_time = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S %Z')

                data.append({
                    'file_name': file_name,
                    'document_page': page_num,
                    'chunk_no': total_chunks,
                    'text': chunk,
                    'model': response.model,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'created_date_time': current_time,
                    'chunk_vector': response.data[0].embedding
                })
                total_chunks += 1

    logger.info(f"Processed {file_name}: {len(pages)} pages, {total_chunks} chunks")
    return pd.DataFrame(data)

def save_dataframe_to_csv(df, file_name):
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    csv_file_path = os.path.join(CSV_OUTPUT_DIR, f"{file_name.replace('.pdf', '')}.csv")
    df.to_csv(csv_file_path, index=False)
    logger.info(f"Saved CSV file: {csv_file_path}")

def process_pdf_files():
    try:
        for file_name in get_pdf_files_from_local():
            try:
                df = process_pdf_and_create_dataframe(file_name)
                if not df.empty:
                    save_dataframe_to_csv(df, file_name)
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
        logger.info("PDF files have been processed and saved as CSV files.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    process_pdf_files()
