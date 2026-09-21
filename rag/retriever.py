import re
from typing import List, Dict, Any, Set


# Small set of common English stop words to ignore during keyword matching
STOP_WORDS: Set[str] = {
    "the", "is", "a", "an", "of", "to", "and", "in", "for",
    "what", "how", "does", "do", "are", "it", "as", "by", "on", "with", "this", "that"
}


class KeywordRetriever:
    """
    A simple, keyword-based passage retriever.
    Scores document chunks by counting matches of important query words.
    No embeddings, vector databases, or complex algorithms required.
    """

    def __init__(self, chunks: List[str]):
        """
        Initialize the retriever with a collection of document chunks.

        :param chunks: List of text passages to search over.
        """
        self.chunks = chunks

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        Extract lowercase alphanumeric words, filtering out stop words.

        :param text: Input query or chunk text.
        :return: List of meaningful keyword tokens.
        """
        # Split text into alphanumeric words and convert to lowercase
        raw_words = re.findall(r"\b[a-zA-Z0-9_\.]+\b", text.lower())
        # Keep non-empty tokens that are not stop words
        return [w for w in raw_words if w not in STOP_WORDS and len(w) > 1]

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Find and return the top-k most relevant chunks for a user query.

        :param query: The user's question or search phrase.
        :param top_k: Maximum number of chunks to return (default: 2).
        :return: List of dicts with {"score": int, "text": str}, sorted by score descending.
                 Only chunks with score > 0 are returned.
        """
        query_keywords = self.extract_keywords(query)

        if not query_keywords:
            return []

        scored_chunks: List[Dict[str, Any]] = []

        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = 0

            # Count occurrences of each query keyword inside the chunk
            for kw in query_keywords:
                # Direct word appearance count
                matches = len(re.findall(r"\b" + re.escape(kw) + r"\b", chunk_lower))
                score += matches

            # Only consider chunks that had at least one keyword match
            if score > 0:
                scored_chunks.append({
                    "score": score,
                    "text": chunk
                })

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        return scored_chunks[:top_k]
