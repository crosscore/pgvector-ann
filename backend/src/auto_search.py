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

CATEGORY_NAME = os.environ.get('CATEGORY_NAME', 'analytics_and_big_data')

required_directories = [
    "../data/log",
    "../data/csv/search_results"
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

def get_db_connection():
    return psycopg2.connect(
        dbname=PGVECTOR_DB_NAME,
        user=PGVECTOR_DB_USER,
        password=PGVECTOR_DB_PASSWORD,
        host=PGVECTOR_DB_HOST,
        port=PGVECTOR_DB_PORT
    )

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
    return response.data[0].embedding

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

def search_similar_chunks(cursor, query_vector, top_n=100):
    vector_type = "halfvec(3072)" if INDEX_TYPE in ["hnsw", "ivfflat"] else "vector(3072)"
    search_query = f"""
    SELECT file_name, document_page, chunk_no, chunk_text,
            (chunk_vector::{vector_type} <#> %s::{vector_type}) AS distance
    FROM document_vectors
    ORDER BY distance ASC
    LIMIT %s;
    """
    cursor.execute(search_query, (query_vector, top_n))
    return cursor.fetchall()

def perform_search(cursor, search_text, file_name, document_page, top_n=100):
    # Measure memory usage before the search
    before_stats = get_container_stats(POSTGRES_CONTAINER_NAME)

    start_time = time.time()
    query_vector = create_embedding(search_text)
    similar_chunks = search_similar_chunks(cursor, query_vector, top_n)
    search_time = round(time.time() - start_time, 4)  # Round to 4 decimal places

    # Measure memory usage after the search
    after_stats = get_container_stats(POSTGRES_CONTAINER_NAME)

    # Find the target rank
    target_rank = next((i + 1 for i, chunk in enumerate(similar_chunks)
                        if chunk[0] == file_name and int(chunk[1]) == int(document_page)), top_n + 1)

    return search_time, similar_chunks, target_rank, before_stats, after_stats

def process_search_csv():
    search_csv = f'../data/csv/search/search_{CATEGORY_NAME}.csv'
    before_search_file = f'../data/csv/search_results/before_search_{CATEGORY_NAME}.csv'
    after_search_file = f'../data/csv/search_results/after_search_{CATEGORY_NAME}.csv'

    df = pd.read_csv(search_csv)
    before_results = []
    after_results = []

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for index, row in df.iterrows():
                search_text = row['search_text']
                file_name = row['file_name']
                document_page = row['document_page']

                search_time, similar_chunks, target_rank, before_stats, after_stats = perform_search(cursor, search_text, file_name, document_page)

                base_stats = {
                    'index_type': INDEX_TYPE,
                    'hnsw_m': HNSW_M,
                    'hnsw_ef_construction': HNSW_EF_CONSTRUCTION,
                    'hnsw_ef_search': HNSW_EF_SEARCH,
                    'ivfflat_lists': IVFFLAT_LISTS,
                    'ivfflat_probes': IVFFLAT_PROBES,
                    'num_of_rows': get_row_count(cursor),
                    'search_time': search_time,
                    'target_rank': int(target_rank),
                    'keyword': search_text,
                    'filepath': file_name,
                    'page': document_page,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S%z'),
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

                logger.info(f"Processed {index + 1}/{len(df)} searches")

    for results, filename in [
        (before_results, before_search_file),
        (after_results, after_search_file)
    ]:
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)
        logger.info(f"Search results saved to {filename}")

def get_row_count(cursor):
    cursor.execute("SELECT COUNT(*) FROM document_vectors;")
    return cursor.fetchone()[0]

if __name__ == "__main__":
    process_search_csv()
