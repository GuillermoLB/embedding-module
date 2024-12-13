from pathlib import Path
import json
import numpy as np
import faiss
import typer
from loguru import logger
from tqdm import tqdm

from embedding_module.config.config import (
    CHUNKED_DATA_DIR,
    EMBEDDED_DATA_DIR,
    INDEXED_DATA_DIR,
    QUERY_DATA_DIR,
)

app = typer.Typer()


def load_embeddings(input_path: Path):
    """Load embeddings and their metadata from a JSONL file."""
    embeddings = []
    metadata = []
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data["embedding"])
            metadata.append(
                {
                    "index": data["index"],
                    "chunk_text": data.get("serialized_text", data.get("chunk_text", "")),
                }
            )
    return embeddings, metadata


def build_faiss_index(embeddings, dimension):
    """Build a FAISS index from the embeddings."""
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)
    return index


def save_faiss_index(index, output_path: Path):
    """Save the FAISS index to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    chunked_data_dir: Path = CHUNKED_DATA_DIR,
    embedded_data_dir: Path = EMBEDDED_DATA_DIR,
    index_file: Path = INDEXED_DATA_DIR / "faiss_index.bin",
    metadata_file: Path = INDEXED_DATA_DIR / "metadata.jsonl",
):
    logger.info("Building FAISS index...")

    # Load all embeddings and metadata
    all_embeddings = []
    all_metadata = []

    for embedding_file in tqdm(list(embedded_data_dir.glob("*.jsonl")), desc="Loading embeddings"):
        embeddings, metadata = load_embeddings(embedding_file)
        all_embeddings.extend(embeddings)
        all_metadata.extend(metadata)

    if not all_embeddings:
        logger.error("No embeddings found.")
        return

    # Determine the dimension of the embeddings
    dimension = len(all_embeddings[0])

    # Build the FAISS index
    index = build_faiss_index(all_embeddings, dimension)

    # Save the FAISS index
    save_faiss_index(index, index_file)

    # Save metadata
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        for meta in all_metadata:
            f.write(json.dumps(meta) + "\n")

    logger.success("FAISS index and metadata built and saved.")


@app.command()
def query_index(
    index_file: Path = INDEXED_DATA_DIR / "faiss_index.bin",
    query_embedding_file: Path = QUERY_DATA_DIR / "query_embedding.json",
    chunked_data_dir: Path = CHUNKED_DATA_DIR,
    embedded_data_dir: Path = EMBEDDED_DATA_DIR,
    top_k: int = 5,
):
    logger.info("Querying FAISS index...")

    # Load the FAISS index
    index = load_faiss_index(index_file)

    # Load the query embedding
    with open(query_embedding_file, "r") as f:
        query_embedding = json.load(f)
    query_vector = np.array([query_embedding["embedding"]]).astype("float32")

    # Retrieve similar documents
    distances, indices = retrieve_similar_documents(index, query_vector, top_k)

    # Load all embeddings to map indices to original data
    all_embeddings = []
    for embedding_file in embedded_data_dir.glob("*.jsonl"):
        embeddings, metadata = load_embeddings(embedding_file)
        all_embeddings.extend(metadata)

    # Load chunked data to retrieve chunk text
    chunked_data = {}
    for chunked_file in chunked_data_dir.glob("*.jsonl"):
        with open(chunked_file, "r") as f:
            for line in f:
                data = json.loads(line)
                chunked_data[data["index"]] = data["chunk_text"]

    logger.info(f"Top {top_k} similar documents:")
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        doc_info = all_embeddings[idx]
        chunk_text = chunked_data.get(doc_info["index"], "Chunk text not found")
        logger.info(f"{i + 1}: Document index {idx} with distance {distance:.4f}")
        logger.info(f"Text: {chunk_text[:200]}...")


if __name__ == "__main__":
    app()
