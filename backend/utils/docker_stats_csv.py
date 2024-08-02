# pgvector-ann/backend/utils/docker_stats_csv.py
import docker
import json
import pandas as pd
from datetime import datetime
import pytz
import asyncio
import logging
import time
import os
from dateutil import parser
from config import (
    SEARCH_CSV_OUTPUT_DIR, INDEX_TYPE, HNSW_M, HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH, IVFFLAT_LISTS, IVFFLAT_PROBES
)

logger = logging.getLogger(__name__)

def get_container_memory_stats(container_name):
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        stats = container.stats(stream=False)
        return stats
    except docker.errors.NotFound:
        logger.error(f"Container {container_name} not found")
    except Exception as e:
        logger.error(f"Error getting container stats: {str(e)}")
        logger.exception("Full traceback:")
    return None

async def collect_memory_stats(container_name, duration, interval=0.1):
    start_time = time.time()
    stats_list = []
    while time.time() - start_time < duration:
        stats = get_container_memory_stats(container_name)
        if stats:
            stats_list.append(stats)
        await asyncio.sleep(interval)
    return stats_list[0] if stats_list else None  # Return only the first stat

def parse_timestamp(timestamp_str):
    try:
        return parser.parse(timestamp_str)
    except Exception as e:
        logger.error(f"Error parsing timestamp: {str(e)}")
        return datetime.now(pytz.utc)

def save_memory_stats_with_extra_info(stats, filename, num_of_rows, search_time, keyword, filepath, page, target_rank):
    try:
        jst = pytz.timezone('Asia/Tokyo')

        memory_stats = stats.get('memory_stats', {})
        flat_stats = {
            'index_type': str(INDEX_TYPE),
            'hnsw_m': int(HNSW_M),
            'hnsw_ef_construction': int(HNSW_EF_CONSTRUCTION),
            'hnsw_ef_search': int(HNSW_EF_SEARCH),
            'ivfflat_lists': int(IVFFLAT_LISTS),
            'ivfflat_probes': int(IVFFLAT_PROBES),
            'num_of_rows': int(num_of_rows),
            'search_time': round(float(search_time), 4),
            'target_rank': int(target_rank) if target_rank is not None else None,
            'keyword': str(keyword),
            'filepath': str(filepath) if filepath else '',
            'page': int(page) if page is not None else None,
            'usage': int(memory_stats.get('usage', 0)),
            'limit': int(memory_stats.get('limit', 0)),
            **{k: int(v) if v is not None else None for k, v in memory_stats.get('stats', {}).items() if isinstance(v, (int, float)) or v is None}
        }

        read_time = parse_timestamp(stats['read'])
        current_time = read_time.astimezone(jst)
        flat_stats['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S%z')

        df = pd.DataFrame([flat_stats])

        columns = [
            'index_type', 'hnsw_m', 'hnsw_ef_construction', 'hnsw_ef_search', 'ivfflat_lists', 'ivfflat_probes',
            'num_of_rows', 'search_time', 'target_rank', 'keyword', 'filepath', 'page', 'timestamp',
            'usage', 'limit'
        ] + [col for col in df.columns if col not in [
            'index_type', 'hnsw_m', 'hnsw_ef_construction', 'hnsw_ef_search', 'ivfflat_lists', 'ivfflat_probes',
            'num_of_rows', 'search_time', 'target_rank', 'keyword', 'filepath', 'page', 'timestamp',
            'usage', 'limit'
        ]]
        df = df[columns]

        # Explicitly set integer columns to int64 dtype, handling None values
        int_columns = ['hnsw_m', 'hnsw_ef_construction', 'hnsw_ef_search', 'ivfflat_lists', 'ivfflat_probes',
                       'num_of_rows', 'target_rank', 'page', 'usage', 'limit'] + [col for col in df.columns if col not in [
                       'index_type', 'search_time', 'keyword', 'filepath', 'timestamp'
        ]]
        for col in int_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Use 'Int64' instead of 'int64'

        os.makedirs(SEARCH_CSV_OUTPUT_DIR, exist_ok=True)
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename, index_col='index', parse_dates=['timestamp'])
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S%z')
            # Ensure integer columns in existing_df are also Int64
            for col in int_columns:
                if col in existing_df.columns:
                    existing_df[col] = pd.to_numeric(existing_df[col], errors='coerce').astype('Int64')
            df = pd.concat([existing_df, df], ignore_index=True)

        df.index.name = 'index'
        df.to_csv(filename, float_format='%.4f')  # Use float_format for search_time
        logger.info(f"Memory stats saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving memory stats: {str(e)}")
        logger.exception("Full traceback:")

def print_memory_stats(stats):
    print("Raw memory stats:")
    print(json.dumps(stats.get('memory_stats', {}), indent=2))
