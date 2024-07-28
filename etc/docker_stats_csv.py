import docker
import json
import pandas as pd
from datetime import datetime
import pytz

def get_container_memory_stats(container_name):
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        stats = container.stats(stream=False)
        return stats
    except docker.errors.NotFound:
        print(f"Container {container_name} not found")
    except Exception as e:
        print(f"Error getting container stats: {str(e)}")
    return None

def print_memory_stats(stats):
    print("Raw memory stats:")
    print(json.dumps(stats.get('memory_stats', {}), indent=2))

def save_memory_stats_to_csv(stats, filename):
    memory_stats = stats.get('memory_stats', {})

    # Flatten the nested dictionary
    flat_stats = {
        'usage': memory_stats.get('usage'),
        'limit': memory_stats.get('limit'),
        **memory_stats.get('stats', {})
    }

    # Get current time in JST (UTC+9)
    jst = pytz.timezone('Asia/Tokyo')
    current_time = datetime.now(jst)

    # Use a consistent timestamp format
    flat_stats['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S%z')

    # Create a DataFrame from the flattened stats
    df = pd.DataFrame([flat_stats])

    # Check if file exists
    try:
        existing_df = pd.read_csv(filename, index_col='index', parse_dates=['timestamp'])
        # Ensure consistent timestamp format for existing data
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    # Ensure the index name is set to 'index'
    df.index.name = 'index'

    # Save to CSV with the index
    df.to_csv(filename)
    print(f"Memory stats saved to {filename}")

if __name__ == "__main__":
    CONTAINER_NAME = "pgvector_db"
    CSV_FILENAME = "docker_memory_stats.csv"

    stats = get_container_memory_stats(CONTAINER_NAME)
    if stats:
        save_memory_stats_to_csv(stats, CSV_FILENAME)
