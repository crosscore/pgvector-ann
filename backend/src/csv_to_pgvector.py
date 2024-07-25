# pgvector-ann/backend/src/csv_to_pgvector.py
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from config import *
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=PGVECTOR_DB_NAME,
            user=PGVECTOR_DB_USER,
            password=PGVECTOR_DB_PASSWORD,
            host=PGVECTOR_DB_HOST,
            port=PGVECTOR_DB_PORT
        )
        logger.info(f"Connected to database: {PGVECTOR_DB_HOST}:{PGVECTOR_DB_PORT}")
        yield conn
    except (KeyError, psycopg2.Error) as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")

def create_table_and_index(cursor):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS document_vectors (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        document_page SMALLINT,
        chunk_no INTEGER,
        chunk_text TEXT,
        model TEXT,
        prompt_tokens INTEGER,
        total_tokens INTEGER,
        created_date_time TIMESTAMPTZ,
        chunk_vector vector(3072)
    );
    """
    cursor.execute(create_table_query)
    logger.info("Table created successfully")

    if INDEX_TYPE == "hnsw":
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS hnsw_document_vectors_chunk_vector_idx ON document_vectors
        USING hnsw ((chunk_vector::halfvec(3072)) halfvec_ip_ops)
        WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
        """
        cursor.execute(create_index_query)
        logger.info("HNSW index created successfully")
    elif INDEX_TYPE == "ivfflat":
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS ivfflat_document_vectors_chunk_vector_idx ON document_vectors
        USING ivfflat ((chunk_vector::halfvec(3072)) halfvec_ip_ops)
        WITH (lists = {IVFFLAT_LISTS});
        """
        cursor.execute(create_index_query)
        logger.info("IVFFlat index created successfully")
    elif INDEX_TYPE == "none":
        logger.info("No index created as per configuration")
    else:
        raise ValueError(f"Unsupported index type: {INDEX_TYPE}")

def process_csv_file(file_path, conn):
    logger.info(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)

    with conn.cursor() as cursor:
        create_table_and_index(cursor)

        insert_query = """
        INSERT INTO document_vectors
        (file_name, document_page, chunk_no, chunk_text, model, prompt_tokens, total_tokens, created_date_time, chunk_vector)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector(3072));
        """

        data = []
        for _, row in df.iterrows():
            embedding = row['chunk_vector']
            if isinstance(embedding, str):
                embedding = eval(embedding)
            if len(embedding) != 3072:
                logger.warning(f"Incorrect vector dimension for row. Expected 3072, got {len(embedding)}. Skipping.")
                continue

            data.append((
                row['file_name'], row['document_page'], row['chunk_no'], row['chunk_text'],
                row['model'], row['prompt_tokens'], row['total_tokens'], row['created_date_time'],
                embedding
            ))

        try:
            execute_batch(cursor, insert_query, data, page_size=BATCH_SIZE)
            conn.commit()
            logger.info(f"Inserted {len(data)} rows into the database")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting batch: {e}")

def process_csv_files():
    try:
        with get_db_connection() as conn:
            for file_name in os.listdir(CSV_OUTPUT_DIR):
                if file_name.endswith('.csv'):
                    csv_file_path = os.path.join(CSV_OUTPUT_DIR, file_name)
                    try:
                        process_csv_file(csv_file_path, conn)
                    except Exception as e:
                        logger.error(f"Error processing {file_name}: {e}")
            logger.info(f"CSV files have been processed and inserted into the database with {INDEX_TYPE.upper()} index.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    process_csv_files()
