from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]  # Adjusted to navigate up two levels
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
CORPUS_DATA_DIR = DATA_DIR / "corpus"
RAW_DATA_DIR = CORPUS_DATA_DIR / "raw"
CHUNKED_DATA_DIR = CORPUS_DATA_DIR / "chunked"
EMBEDDED_DATA_DIR = CORPUS_DATA_DIR / "embedded"
INDEXED_DATA_DIR = CORPUS_DATA_DIR / "indexed"
QUERY_DATA_DIR = DATA_DIR / "queries" / "query"

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
