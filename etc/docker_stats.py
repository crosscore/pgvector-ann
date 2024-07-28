# docker_memory_stats.py
import docker
import json

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

if __name__ == "__main__":
    CONTAINER_NAME = "pgvector_db"
    stats = get_container_memory_stats(CONTAINER_NAME)
    print_memory_stats(stats) # statsの全情報をJSON形式で出力
