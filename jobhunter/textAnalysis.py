import json
import logging
import os
import time
import random
from typing import List, Union

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from different possible locations
load_dotenv()  # Try current directory first
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))  # Try one level up (project root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_embedding_model():
    logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2' into memory...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_openai_api_key():
    """
    Get the OpenAI API key from session state or environment variable.
    Maintained for backwards compatibility with UI elements.
    """
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
        if _is_placeholder_key(api_key):
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and not _is_placeholder_key(api_key):
                return api_key
            return None
        return api_key
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and not _is_placeholder_key(api_key):
        return api_key
    return None

def _is_placeholder_key(api_key):
    """Check if the API key appears to be a placeholder or demo value"""
    if not api_key:
        return True
    placeholders = [
        "your", "api", "key", "here", "demo", "example", "sample", "test", "placeholder",
        "sk-demo", "sk-test", "enter", "insert", "provide"
    ]
    key_lower = api_key.lower()
    for placeholder in placeholders:
        if placeholder in key_lower:
            return True
    if len(api_key) < 20:
        return True
    if not key_lower.startswith("sk-"):
        return True
    return False

def generate_gpt_embedding(text: str) -> List[float]:
    """
    Generate embeddings for input text using local sentence-transformers model.
    """
    if not text or len(text.strip()) == 0:
        return [0.0] * 384

    try:
        model = get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return [float(x) for x in embedding]
    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return [0.0] * 384

def generate_gpt_embeddings_batch(texts: List[str]) -> List[Union[List[float], None]]:
    """
    Generate embeddings for a batch of input texts using local sentence-transformers model.
    """
    if not texts:
        return []

    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in emb] for emb in embeddings]
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
        return [None] * len(texts)

if __name__ == "__main__":
    test_texts = [
        "This is the first sentence for batching.",
        "Here is another sentence, slightly longer.",
        "A third one to test the batch call."
    ]
    batch_embeddings = generate_gpt_embeddings_batch(test_texts)
    
    if batch_embeddings:
        print(f"Received {len(batch_embeddings)} embeddings.")
        for i, emb in enumerate(batch_embeddings):
            if emb:
                print(f"Embedding {i+1} length: {len(emb)}")
            else:
                print(f"Embedding {i+1}: Failed")
    else:
        print("Batch embedding failed entirely.")
