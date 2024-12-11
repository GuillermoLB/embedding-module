from pathlib import Path
import json

import typer
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from embedding_module.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EMBED_MODEL_ID

app = typer.Typer()

def load_chunks(input_path: Path):
    """Load chunks from a JSONL file."""
    with open(input_path, 'r') as f:
        chunks = [json.loads(line) for line in f]
    return chunks

def initialize_model(model_id: str):
    """Initialize the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return tokenizer, model

def generate_embeddings(chunks, tokenizer, model):
    """Generate embeddings for each chunk."""
    embeddings = []
    for chunk in tqdm(chunks, desc="Generating embeddings"):
        inputs = tokenizer(chunk['chunk_text'], return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
        embeddings.append({
            "index": chunk["index"],
            "embedding": embedding.tolist()
        })
    return embeddings

def save_embeddings_to_jsonl(embeddings, output_path: Path):
    """Save embeddings to a JSONL file."""
    with open(output_path, 'w') as f:
        for embedding in embeddings:
            f.write(json.dumps(embedding) + '\n')

@app.command()
def main(
    interim_data_dir: Path = INTERIM_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
):
    logger.info("Starting embedding generation...")

    # Ensure the interim and processed directories exist
    interim_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model and tokenizer
    tokenizer, model = initialize_model(EMBED_MODEL_ID)

    # Process each file in the interim data directory
    for input_path in tqdm(list(interim_data_dir.glob("*.jsonl")), desc="Processing files"):
        logger.info(f"Processing file: {input_path.name}")

        # Load chunks
        chunks = load_chunks(input_path)

        # Generate embeddings
        embeddings = generate_embeddings(chunks, tokenizer, model)

        # Define the output path
        output_path = processed_data_dir / f"{input_path.stem}_embeddings.jsonl"

        # Save embeddings to JSONL file
        save_embeddings_to_jsonl(embeddings, output_path)

    logger.success("Embedding generation complete.")

if __name__ == "__main__":
    app()