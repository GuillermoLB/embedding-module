from pathlib import Path
import json
import numpy as np
import faiss
import typer
from loguru import logger
from tqdm import tqdm

from embedding_module.config import PROCESSED_DATA_DIR, EMBED_MODEL_ID, INDEX_DATA_DIR, QUERY_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def load_embeddings(input_path: Path):
    """Load embeddings from a JSONL file."""
    embeddings = []
    with open(input_path, 'r') as f:
        for line in f:
            embeddings.append(json.loads(line))
    return embeddings

def build_faiss_index(embeddings, dimension):
    """Build a FAISS index from the embeddings."""
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array([embedding['embedding'] for embedding in embeddings]).astype('float32')
    index.add(vectors)
    return index

def save_faiss_index(index, output_path: Path):
    """Save the FAISS index to a file."""
    faiss.write_index(index, str(output_path))

def load_faiss_index(input_path: Path):
    """Load the FAISS index from a file."""
    return faiss.read_index(str(input_path))

def retrieve_similar_documents(index, query_vector, top_k=5):
    """Retrieve the most similar documents to the query vector."""
    distances, indices = index.search(query_vector, top_k)
    return distances, indices

@app.command()
def build_index(
    embeddings_dir: Path = PROCESSED_DATA_DIR,
    index_file: Path = INDEX_DATA_DIR / "faiss_index.bin",
):
    logger.info("Building FAISS index...")

    # Load all embeddings
    all_embeddings = []
    for embedding_file in tqdm(list(embeddings_dir.glob("*.jsonl")), desc="Loading embeddings"):
        embeddings = load_embeddings(embedding_file)
        all_embeddings.extend(embeddings)

    # Check if there are any embeddings
    if not all_embeddings:
        logger.error("No embeddings found.")
        return

    # Determine the dimension of the embeddings
    dimension = len(all_embeddings[0]['embedding'])

    # Build the FAISS index
    index = build_faiss_index(all_embeddings, dimension)

    # Save the FAISS index
    save_faiss_index(index, index_file)

    logger.success("FAISS index built and saved.")

@app.command()
def query_index(
    index_file: Path = INDEX_DATA_DIR / "faiss_index.bin",
    query_embedding_file: Path = QUERY_DATA_DIR / "query_embedding.json",
    embeddings_dir: Path = INTERIM_DATA_DIR,
    top_k: int = 5,
):
    logger.info("Querying FAISS index...")

    # Load the FAISS index
    index = load_faiss_index(index_file)

    # Load the query embedding
    with open(query_embedding_file, 'r') as f:
        query_embedding = json.load(f)
    query_vector = np.array([query_embedding['embedding']]).astype('float32')

    # Retrieve similar documents
    distances, indices = retrieve_similar_documents(index, query_vector, top_k)

    # Load all embeddings to map indices to original data
    all_embeddings = []
    for embedding_file in embeddings_dir.glob("*.jsonl"):
        embeddings = load_embeddings(embedding_file)
        all_embeddings.extend(embeddings)

    logger.info(f"Top {top_k} similar documents:")
    for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
        document_info = all_embeddings[index]
        logger.info(f"{i+1}: Document index {index} with distance {distance}")
        logger.info(f"Text: {document_info['chunk_text'][:200]}...")  # Display first 200 characters of the text

if __name__ == "__main__":
    app()