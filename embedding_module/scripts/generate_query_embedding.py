from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModel
import torch

from embedding_module.config.config import EMBED_MODEL_ID, PROCESSED_DATA_DIR

def initialize_model(model_id: str):
    """Initialize the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return tokenizer, model

def generate_embedding(text: str, tokenizer, model):
    """Generate an embedding for the given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return embedding

def save_embedding(embedding, output_path: Path):
    """Save the embedding to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump({"embedding": embedding.tolist()}, f)

def main(query_text: str, output_path: Path):
    # Initialize model and tokenizer
    tokenizer, model = initialize_model(EMBED_MODEL_ID)

    # Generate embedding
    embedding = generate_embedding(query_text, tokenizer, model)

    # Save embedding to JSON file
    save_embedding(embedding, output_path)

if __name__ == "__main__":
    query_text = "Example query text to generate embedding"
    output_path = PROCESSED_DATA_DIR / "query_embedding.json"
    main(query_text, output_path)