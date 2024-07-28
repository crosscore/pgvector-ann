# pgvector-ann/backend/utils/docker_stats_csv.py
import docker
import json
import pandas as pd
from datetime import datetime
import pytz
import asyncio
import logging
import time

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
    return None

async def collect_memory_stats(container_name, duration, interval=0.1):
    start_time = time.time()
    stats_list = []
    while time.time() - start_time < duration:
        stats = get_container_memory_stats(container_name)
        if stats:
            stats_list.append(stats)
        await asyncio.sleep(interval)
    return stats_list

def save_memory_stats_with_extra_info(stats_list, filename, index_type, search_time, keyword):
    df_list = []
    jst = pytz.timezone('Asia/Tokyo')

    for stats in stats_list:
        memory_stats = stats.get('memory_stats', {})
        flat_stats = {
            'usage': memory_stats.get('usage'),
            'limit': memory_stats.get('limit'),
            **memory_stats.get('stats', {}),
            'index_type': index_type if index_type in ['hnsw', 'ivfflat'] else 'None',
            'search_time': search_time,
            'keyword': keyword
        }

        # Add timestamp
        current_time = datetime.fromtimestamp(stats['read'] / 1e9, tz=jst)  # Converting nanoseconds to seconds
        flat_stats['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S%z')

        df_list.append(pd.DataFrame([flat_stats]))

    df = pd.concat(df_list, ignore_index=True)

    # Reorder columns
    columns = ['index_type', 'search_time', 'keyword', 'timestamp'] + [col for col in df.columns if col not in ['index_type', 'search_time', 'keyword', 'timestamp']]
    df = df[columns]

    # Check if file exists and append
    try:
        existing_df = pd.read_csv(filename, index_col='index', parse_dates=['timestamp'])
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.index.name = 'index'
    df.to_csv(filename)
    logger.info(f"Memory stats saved to {filename}")

def print_memory_stats(stats):
    print("Raw memory stats:")
    print(json.dumps(stats.get('memory_stats', {}), indent=2))

if __name__ == "__main__":
    CONTAINER_NAME = "pgvector_db"
    CSV_FILENAME = "docker_memory_stats.csv"

    stats = get_container_memory_stats(CONTAINER_NAME)
    if stats:
        save_memory_stats_with_extra_info([stats], CSV_FILENAME, 'None', 0, 'test')
