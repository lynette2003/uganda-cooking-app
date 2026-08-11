"""
train_llm.py

Builds the semantic retrieval index for the RAG system.

This does NOT fine-tune GPT-2.

Instead, it:
1. Loads the processed recipe knowledge base.
2. Converts each recipe into an embedding.
3. Saves the embeddings for fast retrieval.

Run:

    python train_llm.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


KNOWLEDGE_BASE = Path(
    "data/processed/recipe_knowledge_base.json"
)

EMBEDDINGS_FILE = Path(
    "data/processed/recipe_embeddings.npy"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_knowledge_base():
    """Load processed recipes."""

    if not KNOWLEDGE_BASE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE}\n"
            "Run data_preprocessing.py first."
        )

    with open(
        KNOWLEDGE_BASE,
        "r",
        encoding="utf-8"
    ) as file:

        recipes = json.load(file)

    if not recipes:
        raise ValueError(
            "The recipe knowledge base is empty."
        )

    return recipes


def build_embeddings():

    logger.info(
        "Loading recipe knowledge base..."
    )

    recipes = load_knowledge_base()

    logger.info(
        "Loading embedding model: %s",
        MODEL_NAME
    )

    model = SentenceTransformer(MODEL_NAME)

    documents = [
        recipe["search_text"]
        for recipe in recipes
    ]

    logger.info(
        "Creating embeddings for %d recipes...",
        len(documents)
    )

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    EMBEDDINGS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    logger.info(
        "Embeddings saved to %s",
        EMBEDDINGS_FILE
    )

    logger.info(
        "Embedding shape: %s",
        embeddings.shape
    )

    return embeddings


if __name__ == "__main__":

    try:
        build_embeddings()

        print()
        print("=" * 60)
        print("RAG INDEX CREATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Knowledge base: {KNOWLEDGE_BASE}")
        print(f"Embeddings:     {EMBEDDINGS_FILE}")
        print()

    except Exception as error:

        logger.error(
            "Failed to build RAG index: %s",
            error
        )

        raise
