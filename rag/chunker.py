from typing import List


class TextChunker:
    """
    Splits continuous document text into smaller, overlapping chunks.
    
    Why Chunking is Needed:
    Large language models have finite context windows, and passing an entire
    corpus into a prompt is inefficient or impossible. Chunking breaks documents
    into bite-sized passages.

    Why Overlap Matters:
    If a key fact or sentence spans right across the boundary between two chunks,
    a pure split without overlap could break the context in half. The overlap
    ensures that boundary information is preserved in at least one full chunk.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize the text chunker.

        :param chunk_size: Number of characters per chunk (default: 500).
        :param overlap: Number of overlapping characters between adjacent chunks (default: 100).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Slice text into chunks with the specified character size and overlap.

        :param text: Raw text to be chunked.
        :return: List of string chunks.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        chunks: List[str] = []
        start_idx = 0
        text_length = len(cleaned_text)

        # Slide a window across the text: step size is (chunk_size - overlap)
        step = self.chunk_size - self.overlap

        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            chunk = cleaned_text[start_idx:end_idx].strip()
            
            if chunk:
                chunks.append(chunk)

            # If we've reached the end of the text, stop
            if end_idx >= text_length:
                break

            start_idx += step

        return chunks
