from pathlib import Path
import json

import typer
from loguru import logger
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

from embedding_module.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64

app = typer.Typer()

def convert_document(input_path: Path):
    """Convert the document using DocumentConverter."""
    converter = DocumentConverter()
    result = converter.convert(input_path)
    return result.document

def initialize_tokenizer(model_id: str):
    """Initialize the tokenizer."""
    return AutoTokenizer.from_pretrained(model_id)

def chunk_document(doc, tokenizer, max_tokens: int):
    """Chunk the document using HybridChunker."""
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    chunk_iter = chunker.chunk(dl_doc=doc)
    return list(chunk_iter)

def serialize_chunks(chunks, tokenizer, chunker):
    """Serialize chunks and return a list of chunk information."""
    chunk_data = []
    for i, chunk in enumerate(chunks):
        txt_tokens = len(tokenizer.tokenize(chunk.text, max_length=None))
        ser_txt = chunker.serialize(chunk=chunk)
        ser_tokens = len(tokenizer.tokenize(ser_txt, max_length=None))
        
        chunk_info = {
            "index": i,
            "chunk_text": chunk.text,
            "chunk_text_tokens": txt_tokens,
            "serialized_text": ser_txt,
            "serialized_text_tokens": ser_tokens
        }
        chunk_data.append(chunk_info)
    return chunk_data

def save_chunks_to_jsonl(chunk_data, output_path: Path):
    """Save chunk data to a JSONL file."""
    with open(output_path, 'w') as f:
        for chunk_info in chunk_data:
            f.write(json.dumps(chunk_info) + '\n')

@app.command()
def main(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_file: Path = PROCESSED_DATA_DIR / "corpus.jsonl",
):
    logger.info("Processing dataset...")

    # Initialize the tokenizer
    tokenizer = initialize_tokenizer(EMBED_MODEL_ID)
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS)

    all_chunk_data = []

    # Process each file in the raw data directory
    for input_path in raw_data_dir.glob("*.pdf"):
        logger.info(f"Processing file: {input_path.name}")
        
        # Convert the document
        doc = convert_document(input_path)
        
        # Chunk the document
        chunks = chunk_document(doc, tokenizer, MAX_TOKENS)
        
        # Serialize chunks
        chunk_data = serialize_chunks(chunks, tokenizer, chunker)
        
        all_chunk_data.extend(chunk_data)
    
    # Save all chunks to a single JSONL file
    save_chunks_to_jsonl(all_chunk_data, processed_data_file)
    
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()