# pgvector-ann/backend/src/auto_search.py
import os
import pandas as pd
import psycopg2
import time
from config import *
from openai import AzureOpenAI, OpenAI
import logging
from datetime import datetime
import docker
import re

CATEGORY_NAME = os.environ.get('CATEGORY_NAME', 'analytics_and_big_data')

required_directories = [
    "../data/log",
    "../data/search_results_csv"
]
for directory in required_directories:
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(filename="../data/log/auto_search.log", level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

if ENABLE_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("Using OpenAI API for embeddings")
else:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    logger.info("Using Azure OpenAI API for embeddings")

docker_client = docker.from_env()

def sanitize_table_name(name):
    # Convert to lowercase first
    name = name.lower()
    # Check if the name is a reserved word
    if name in ['all', 'order', 'user', 'table', 'select', 'where', 'from', 'group', 'by']:
        name = f"t_{name}"
    # Remove any character that isn't alphanumeric or underscore
    sanitized = re.sub(r'\W+', '_', name)
    # Ensure the name starts with a letter
    if not sanitized[0].isalpha():
        sanitized = "t_" + sanitized
    return sanitized

def get_db_connection():
    return psycopg2.connect(
        dbname=PGVECTOR_DB_NAME,
        user=PGVECTOR_DB_USER,
        password=PGVECTOR_DB_PASSWORD,
        host=PGVECTOR_DB_HOST,
        port=PGVECTOR_DB_PORT
    )

def create_embedding(text):
    try:
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
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        raise

def get_container_stats(container_name):
    try:
        container = docker_client.containers.get(container_name)
        stats = container.stats(stream=False)
        return stats
    except docker.errors.NotFound:
        logger.error(f"Container {container_name} not found")
    except Exception as e:
        logger.error(f"Error getting container stats: {str(e)}")
    return None

def search_similar_chunks(cursor, query_vector, table_name, top_n=100):
    sanitized_table_name = sanitize_table_name(table_name)
    vector_type = "halfvec(3072)" if INDEX_TYPE in ["hnsw", "ivfflat"] else "vector(3072)"
    search_query = f"""
    SELECT file_name, document_page, chunk_no, chunk_text,
            (chunk_vector::{vector_type} <#> %s::{vector_type}) AS distance
    FROM {sanitized_table_name}
    ORDER BY distance ASC
    LIMIT %s;
    """
    try:
        cursor.execute(search_query, (query_vector, top_n))
        return cursor.fetchall()
    except psycopg2.Error as e:
        logger.error(f"Database error during search: {str(e)}")
        raise

def perform_search(cursor, search_text, file_name, document_page, table_name, top_n=100):
    before_stats = get_container_stats(POSTGRES_CONTAINER_NAME)

    start_time = time.time()
    query_vector = create_embedding(search_text)
    similar_chunks = search_similar_chunks(cursor, query_vector, table_name, top_n)
    search_time = round(time.time() - start_time, 4)

    after_stats = get_container_stats(POSTGRES_CONTAINER_NAME)

    target_rank = next((i + 1 for i, chunk in enumerate(similar_chunks)
                        if chunk[0] == file_name and int(chunk[1]) == int(document_page)), top_n + 1)

    return search_time, similar_chunks, target_rank, before_stats, after_stats

def process_search_csv():
    search_csv = f'../data/search_csv/search_{CATEGORY_NAME}.csv'
    before_search_file = f'../data/search_results_csv/before_search_{CATEGORY_NAME}.csv'
    after_search_file = f'../data/search_results_csv/after_search_{CATEGORY_NAME}.csv'

    if not os.path.exists(search_csv):
        logger.error(f"Search CSV file not found: {search_csv}")
        return

    df = pd.read_csv(search_csv)
    before_results = []
    after_results = []

    table_name = sanitize_table_name(CATEGORY_NAME)

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for index, row in df.iterrows():
                search_text = row['search_text']
                file_name = row['file_name']
                document_page = row['document_page']

                try:
                    search_time, similar_chunks, target_rank, before_stats, after_stats = perform_search(cursor, search_text, file_name, document_page, table_name)

                    base_stats = {
                        'index_type': INDEX_TYPE,
                        'hnsw_m': HNSW_M,
                        'hnsw_ef_construction': HNSW_EF_CONSTRUCTION,
                        'hnsw_ef_search': HNSW_EF_SEARCH,
                        'ivfflat_lists': IVFFLAT_LISTS,
                        'ivfflat_probes': IVFFLAT_PROBES,
                        'num_of_rows': get_row_count(cursor, table_name),
                        'search_time': search_time,
                        'target_rank': int(target_rank),
                        'keyword': search_text,
                        'filepath': file_name,
                        'page': document_page,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S%z'),
                        'category': CATEGORY_NAME,
                    }

                    for stats, results in [(before_stats, before_results), (after_stats, after_results)]:
                        if stats:
                            memory_stats = stats.get('memory_stats', {})
                            current_stats = base_stats.copy()
                            current_stats.update({
                                'usage': memory_stats.get('usage', 0),
                                'limit': memory_stats.get('limit', 0),
                                **{k: v for k, v in memory_stats.get('stats', {}).items() if isinstance(v, (int, float))}
                            })
                            results.append(current_stats)

                    logger.info(f"Processed {index + 1}/{len(df)} searches for category {CATEGORY_NAME}")
                except Exception as e:
                    logger.error(f"Error processing search for index {index}: {str(e)}")

    for results, filename in [
        (before_results, before_search_file),
        (after_results, after_search_file)
    ]:
        results_df = pd.DataFrame(results)
        
        if os.path.exists(filename):
            results_df.to_csv(filename, mode='a', header=False, index=False)
            logger.info(f"Appended search results to existing file: {filename}")
        else:
            results_df.to_csv(filename, index=False)
            logger.info(f"Created new file with search results: {filename}")

def get_row_count(cursor, table_name):
    sanitized_table_name = sanitize_table_name(table_name)
    cursor.execute(f"SELECT COUNT(*) FROM {sanitized_table_name};")
    return cursor.fetchone()[0]

def main():
    logger.info(f"Starting auto_search for category: {CATEGORY_NAME}")
    sanitized_table_name = sanitize_table_name(CATEGORY_NAME)
    logger.info(f"Using table: {sanitized_table_name}")
    
    try:
        process_search_csv()
        logger.info(f"Completed auto_search for category: {CATEGORY_NAME}")
    except Exception as e:
        logger.error(f"Error during auto_search for category {CATEGORY_NAME}: {str(e)}")
        raise

if __name__ == "__main__":
    main()
