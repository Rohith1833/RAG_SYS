from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    A lightweight, dedicated text embedding component for RAG V2.

    -------------------------------------------------------------------------
    WHAT IS AN EMBEDDING?
    An embedding is a numerical vector (a list of numbers) that represents
    the semantic meaning of a piece of text.

    WHY DOES TEXT BECOME A VECTOR?
    Computers cannot understand human language directly, nor can they compare
    the conceptual meaning of words using simple string comparisons. By mapping
    text into a continuous multi-dimensional space, words or passages become
    geometric points or arrows (vectors).

    WHY DO SIMILAR MEANINGS PRODUCE SIMILAR VECTORS?
    Embedding models (like MiniLM) are trained using contrastive objectives on
    billions of sentence pairs. The model learns to position sentences with
    similar meanings close together in vector space (high cosine similarity),
    even when they share zero exact matching keywords.
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Union[str, torch.device, None] = None
    ):
        """
        Initialize the sentence embedding model.

        :param model_name: HuggingFace sentence-transformer model identifier
                           (default: 'all-MiniLM-L6-v2', ~80MB, 384 dimensions).
        :param device: Hardware device for inference ('cuda' or 'cpu').
                       If None, automatically selects CUDA if available.
        """
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        # Load the pre-trained embedding model
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        # Determine embedding dimension (typically 384 for MiniLM-L6-v2)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()


    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single text string into a 1D embedding vector.

        :param text: Input string to embed.
        :return: 1D numpy array of shape (embedding_dim,) with normalized float32 values.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty or whitespace-only text.")

        # normalize_embeddings=True ensures vectors have unit length (L2 norm = 1.0),
        # which makes cosine similarity equivalent to a simple dot product.
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a batch of text passages into a 2D matrix of embedding vectors.

        :param texts: List of strings to embed.
        :return: 2D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )

        return embeddings.astype(np.float32)
