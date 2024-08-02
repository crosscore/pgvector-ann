import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

json_dir = "./json/"
os.makedirs(json_dir, exist_ok=True)

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

text = "Hello, world!"
response = client.embeddings.create(
    input=text,
    model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
)

response_dict = response.model_dump()
response_json = json.dumps(response_dict, indent=2)
with open(os.path.join(json_dir, "no_langchain_embedding.json"), "w") as f:
    f.write(response_json)
print(f"Response has been saved to {os.path.join(json_dir, 'no_langchain_embedding.json')}")

embedding = response.data[0].embedding
print(f"Embedding for '{text}':")
print(embedding[:5])
print(f"Embedding dimension: {len(embedding)}")

output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame({
    'dimension': range(1, len(embedding) + 1),
    'value': embedding
})
csv_filename = os.path.join(output_dir, "embedding_vector_no_langchain.csv")
df.to_csv(csv_filename, index=False)
print(f"Embedding vector has been saved to {csv_filename}")