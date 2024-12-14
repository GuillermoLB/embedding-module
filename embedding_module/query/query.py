# embedding_module/query/query.py

import json
from pathlib import Path
import numpy as np
from embedding_module.config.config import (
    INDEXED_DATA_DIR,
    EMBED_MODEL_ID,
)
from embedding_module.embedding.embedder import initialize_model, generate_embedding
from embedding_module.indexing.indexer import (
    load_faiss_index,
    retrieve_similar_documents,
)

def load_metadata(metadata_file: Path):
    """Load metadata from a JSONL file."""
    metadata = []
    with open(metadata_file, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

def query(question: str, top_k: int = 5):
    """Perform a query and return the top_k results."""
    # Initialize tokenizer and model
    tokenizer, model = initialize_model(EMBED_MODEL_ID)
    
    # Generate embedding for the query
    query_embedding = generate_embedding(question, tokenizer, model)
    query_vector = np.array([query_embedding]).astype('float32')
    
    # Load the FAISS index
    index = load_faiss_index(INDEXED_DATA_DIR / "faiss_index.bin")
    
    # Retrieve similar documents
    distances, indices = retrieve_similar_documents(index, query_vector, top_k)
    
    # Load metadata to map indices to original data
    metadata = load_metadata(INDEXED_DATA_DIR / "metadata.jsonl")
    
    # Collect the results
    results = []
    for i, idx in enumerate(indices[0]):
        doc_info = metadata[idx]
        results.append({
            'rank': i + 1,
            'index': idx,
            'distance': float(distances[0][i]),
            'chunk_text': doc_info.get('chunk_text', ''),
        })
    return results