# pgvector-ann/backend/src/csv_to_pgvector.py
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from config import *
import logging
from contextlib import contextmanager

logging.basicConfig(filename="/app/data/log/csv_to_pgvector.log", level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    conn = None
    try:
        with psycopg2.connect(
            dbname=PGVECTOR_DB_NAME,
            user=PGVECTOR_DB_USER,
            password=PGVECTOR_DB_PASSWORD,
            host=PGVECTOR_DB_HOST,
            port=PGVECTOR_DB_PORT
        ) as conn:
            logger.info(f"Connected to database: {PGVECTOR_DB_HOST}:{PGVECTOR_DB_PORT}")
            yield conn
    except (KeyError, psycopg2.Error) as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        logger.info("Database connection closed")

def create_table_and_index(cursor, table_name):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        document_page SMALLINT,
        chunk_no INTEGER,
        chunk_text TEXT,
        model TEXT,
        prompt_tokens INTEGER,
        total_tokens INTEGER,
        created_date_time TIMESTAMPTZ,
        chunk_vector vector(3072),
        business_category TEXT
    );
    """
    cursor.execute(create_table_query)
    logger.info(f"Table {table_name} created successfully")

    if INDEX_TYPE == "hnsw":
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS hnsw_{table_name}_chunk_vector_idx ON {table_name}
        USING hnsw ((chunk_vector::halfvec(3072)) halfvec_ip_ops)
        WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
        """
        cursor.execute(create_index_query)
        logger.info(f"HNSW index created successfully for {table_name} with parameters: m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}")
    elif INDEX_TYPE == "ivfflat":
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS ivfflat_{table_name}_chunk_vector_idx ON {table_name}
        USING ivfflat ((chunk_vector::halfvec(3072)) halfvec_ip_ops)
        WITH (lists = {IVFFLAT_LISTS});
        """
        cursor.execute(create_index_query)
        logger.info(f"IVFFlat index created successfully for {table_name} with parameter: lists = {IVFFLAT_LISTS}")
    elif INDEX_TYPE == "none":
        logger.info(f"No index created for {table_name} as per configuration")
    else:
        raise ValueError(f"Unsupported index type: {INDEX_TYPE}")

def process_csv_file(file_path, cursor, table_name):
    logger.info(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)

    insert_query = f"""
    INSERT INTO {table_name}
    (file_name, document_page, chunk_no, chunk_text, model, prompt_tokens, total_tokens, created_date_time, chunk_vector, business_category)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector(3072), %s);
    """

    data = []
    for _, row in df.iterrows():
        embedding = row['chunk_vector']
        if isinstance(embedding, str):
            embedding = eval(embedding)
        if len(embedding) != 3072:
            logger.warning(f"Incorrect vector dimension for row. Expected 3072, got {len(embedding)}. Skipping.")
            continue

        business_category = row.get('business_category', os.path.basename(os.path.dirname(file_path)))

        data.append((
            row['file_name'], row['document_page'], row['chunk_no'], row['chunk_text'],
            row['model'], row['prompt_tokens'], row['total_tokens'], row['created_date_time'],
            embedding, business_category
        ))

    try:
        execute_batch(cursor, insert_query, data, page_size=BATCH_SIZE)
        logger.info(f"Inserted {len(data)} rows into the {table_name} table")
    except Exception as e:
        logger.error(f"Error inserting batch into {table_name}: {e}")
        raise

def process_category_csv_files(conn):
    logger.info("Processing CSV files by category")
    for category in PROCESS_CATEGORIES.split(','):
        category = category.strip()
        category_dir = os.path.join(CSV_OUTPUT_DIR, category)
        
        if not os.path.isdir(category_dir):
            logger.warning(f"Category directory not found: {category_dir}")
            continue

        table_name = f"document_vectors_{category}"
        
        with conn.cursor() as cursor:
            create_table_and_index(cursor, table_name)
            
            for file in os.listdir(category_dir):
                if file.endswith('.csv'):
                    csv_file_path = os.path.join(category_dir, file)
                    try:
                        process_csv_file(csv_file_path, cursor, table_name)
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Error processing {csv_file_path}: {e}")

def process_all_csv(conn):
    logger.info("Processing all.csv file")
    all_csv_path = os.path.join(CSV_OUTPUT_DIR, "all", "all.csv")
    if os.path.exists(all_csv_path):
        with conn.cursor() as cursor:
            create_table_and_index(cursor, "document_vectors")
            process_csv_file(all_csv_path, cursor, "document_vectors")
            conn.commit()
    else:
        logger.error(f"all.csv file not found at {all_csv_path}")
        raise FileNotFoundError(f"all.csv file not found at {all_csv_path}")

def process_csv_files():
    ENABLE_ALL_CSV = os.getenv("ENABLE_ALL_CSV", "false").lower() == "true"
    ENABLE_CATEGORY_TABLES = os.getenv("ENABLE_CATEGORY_TABLES", "false").lower() == "true"

    logger.info(f"Processing with ENABLE_ALL_CSV: {ENABLE_ALL_CSV}")
    logger.info(f"Processing with ENABLE_CATEGORY_TABLES: {ENABLE_CATEGORY_TABLES}")

    if not ENABLE_ALL_CSV and not ENABLE_CATEGORY_TABLES:
        error_msg = "Neither ENABLE_ALL_CSV nor ENABLE_CATEGORY_TABLES is set to true. Please enable at least one processing mode."
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        with get_db_connection() as conn:
            if ENABLE_ALL_CSV:
                process_all_csv(conn)
            if ENABLE_CATEGORY_TABLES:
                process_category_csv_files(conn)

            logger.info(f"CSV files have been processed and inserted into the database with {INDEX_TYPE.upper()} index.")
            if INDEX_TYPE == "hnsw":
                logger.info(f"HNSW index parameters: m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}")
            elif INDEX_TYPE == "ivfflat":
                logger.info(f"IVFFlat index parameter: lists = {IVFFLAT_LISTS}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise

if __name__ == "__main__":
    try:
        process_csv_files()
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        exit(1)