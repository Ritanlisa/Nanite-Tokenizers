import logging
import os
from typing import Callable, Iterable, Optional, Tuple

import chromadb
import faiss
import numpy as np
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore

import config

logger = logging.getLogger(__name__)


def get_vector_store(
    store_type: str,
    persist_dir: str,
    embed_dim: int,
    embed_model=None,
    sample_texts: Optional[Iterable[str]] = None,
    index_builder: Optional[Callable[[int], faiss.Index]] = None,
):
    if store_type == "chroma":
        try:
            chroma_client = chromadb.PersistentClient(path=os.path.join(persist_dir, "chroma"))
            chroma_collection = chroma_client.get_or_create_collection("llama_index")
            return ChromaVectorStore(chroma_collection=chroma_collection)
        except Exception as exc:
            logger.error("Chroma init failed, falling back to memory Faiss: %s", exc)
            return _create_memory_faiss(embed_dim, fallback=True)

    if store_type == "faiss":
        faiss_path = os.path.join(persist_dir, "faiss.index")
        try:
            if os.path.exists(faiss_path):
                index = faiss.read_index(faiss_path)
                logger.info("Loaded Faiss index from file")
            else:
                if index_builder is not None:
                    index = index_builder(embed_dim)
                else:
                    index = _create_faiss_index(embed_dim, embed_model, sample_texts)
                logger.info("Created new Faiss index")
            return FaissVectorStore(faiss_index=index)
        except Exception as exc:
            logger.error("Faiss init failed, falling back to memory Faiss: %s", exc)
            return _create_memory_faiss(embed_dim, fallback=True)

    raise ValueError(f"Unsupported vector store type: {store_type}")


def _create_memory_faiss(embed_dim: int, fallback: bool = False):
    index = faiss.IndexFlatL2(embed_dim)
    vector_store = FaissVectorStore(faiss_index=index)
    if fallback:
        setattr(vector_store, "fallback", True)
    return vector_store


def _parse_ivf_config(index_type: str) -> Tuple[int, str]:
    # Example: "IVF100,Flat" -> nlist=100, metric="Flat"
    parts = index_type.split(",")
    nlist = int(parts[0].replace("IVF", "").strip())
    metric = parts[1].strip() if len(parts) > 1 else "Flat"
    return nlist, metric


def _create_faiss_index(
    embed_dim: int,
    embed_model=None,
    sample_texts: Optional[Iterable[str]] = None,
) -> faiss.Index:
    index_type = config.settings.FAISS_INDEX_TYPE
    if index_type.startswith("IVF"):
        nlist, metric = _parse_ivf_config(index_type)
        quantizer = faiss.IndexFlatL2(embed_dim)
        index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist)
        train_data = None
        if embed_model and sample_texts:
            texts = list(sample_texts)
            sample_size = min(len(texts), config.settings.SAMPLE_FOR_TRAINING)
            if sample_size > 0:
                texts = texts[:sample_size]
                sample_embeds = np.array(
                    [embed_model.get_text_embedding(text) for text in texts],
                    dtype="float32",
                )
                train_data = sample_embeds
        if train_data is None or len(train_data) < max(1, nlist):
            if train_data is not None:
                logger.warning(
                    "Insufficient IVF training samples (%s), padding with random vectors",
                    len(train_data),
                )
            # Train with random vectors if real samples are unavailable or insufficient.
            min_samples = max(1000, nlist * 5)
            random_data = np.random.rand(min_samples, embed_dim).astype("float32")
            train_data = (
                random_data
                if train_data is None
                else np.concatenate([train_data, random_data], axis=0)
            )
        index.train(train_data)
        return index
    return faiss.IndexFlatL2(embed_dim)
