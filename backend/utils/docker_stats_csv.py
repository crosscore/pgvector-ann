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
from config import SEARCH_CSV_OUTPUT_DIR, INDEX_TYPE

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

def save_memory_stats_with_extra_info(stats, filename, num_of_rows, search_time, keyword, operation):
    try:
        jst = pytz.timezone('Asia/Tokyo')

        memory_stats = stats.get('memory_stats', {})
        flat_stats = {
            'usage': int(memory_stats.get('usage', 0)),
            'limit': int(memory_stats.get('limit', 0)),
            **{k: int(v) for k, v in memory_stats.get('stats', {}).items() if isinstance(v, (int, float))},
            'index_type': str(INDEX_TYPE),
            'num_of_rows': int(num_of_rows),
            'search_time': round(float(search_time), 6),
            'keyword': str(keyword),
            'operation': str(operation)
        }

        read_time = parse_timestamp(stats['read'])
        current_time = read_time.astimezone(jst)
        flat_stats['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S%z')

        df = pd.DataFrame([flat_stats])

        columns = ['index_type', 'num_of_rows', 'search_time', 'keyword', 'operation', 'timestamp'] + [col for col in df.columns if col not in ['index_type', 'num_of_rows', 'search_time', 'keyword', 'operation', 'timestamp']]
        df = df[columns]

        # Explicitly set integer columns to int64 dtype
        int_columns = ['usage', 'limit', 'num_of_rows'] + [col for col in df.columns if col not in ['index_type', 'search_time', 'keyword', 'operation', 'timestamp']]
        for col in int_columns:
            df[col] = df[col].astype('int64')

        os.makedirs(SEARCH_CSV_OUTPUT_DIR, exist_ok=True)
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename, index_col='index', parse_dates=['timestamp'])
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S%z')
            # Ensure integer columns in existing_df are also int64
            for col in int_columns:
                if col in existing_df.columns:
                    existing_df[col] = existing_df[col].astype('int64')
            df = pd.concat([existing_df, df], ignore_index=True)

        df.index.name = 'index'
        df.to_csv(filename, float_format='%.6f')  # Use float_format for search_time
        logger.info(f"Memory stats saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving memory stats: {str(e)}")
        logger.exception("Full traceback:")

def print_memory_stats(stats):
    print("Raw memory stats:")
    print(json.dumps(stats.get('memory_stats', {}), indent=2))
