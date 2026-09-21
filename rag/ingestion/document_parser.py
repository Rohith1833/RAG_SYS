from pathlib import Path
from typing import List, Dict, Any
import pypdf


class DocumentParser:
    """
    A lightweight, multi-format document parser for the Ultron RAG system.
    
    Responsibilities:
    1. Validate document file types (.txt, .md, .pdf).
    2. Extract raw text from text-based documents (.txt, .md).
    3. Extract text page-by-page from PDF documents (.pdf) using pypdf to preserve page numbers.
    4. Provide structured document segments ready for text chunking.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    @classmethod
    def is_supported(cls, file_path: str | Path) -> bool:
        """Check whether the file extension is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def parse(cls, file_path: str | Path) -> List[Dict[str, Any]]:
        """
        Parse a document and return a list of text segments with page numbers.

        :param file_path: Absolute or relative path to the document file.
        :return: List of dicts:
                 [{"page": int | None, "text": str, "char_count": int}]
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document does not exist at: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            supported = ", ".join(cls.SUPPORTED_EXTENSIONS)
            raise ValueError(
                f"Unsupported file format '{ext}'. Ultron currently supports: {supported}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"The uploaded document '{path.name}' is completely empty (0 bytes).")

        if ext in {".txt", ".md"}:
            return cls._parse_text_file(path)
        elif ext == ".pdf":
            return cls._parse_pdf_file(path)
        else:
            raise ValueError(f"Unhandled file extension: {ext}")

    @classmethod
    def _parse_text_file(cls, path: Path) -> List[Dict[str, Any]]:
        """Extract text from plain text or markdown files."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
        except Exception as err:
            raise RuntimeError(f"Failed to read text file '{path.name}': {err}")

        if not content:
            raise ValueError(f"Document '{path.name}' contains no readable text content.")

        return [{
            "page": None,
            "text": content,
            "char_count": len(content)
        }]

    @classmethod
    def _parse_pdf_file(cls, path: Path) -> List[Dict[str, Any]]:
        """Extract text page-by-page from PDF files using pypdf."""
        segments: List[Dict[str, Any]] = []

        try:
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)

            if total_pages == 0:
                raise ValueError(f"PDF '{path.name}' contains 0 pages.")

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                cleaned = page_text.strip()
                if cleaned:
                    segments.append({
                        "page": page_num,
                        "text": cleaned,
                        "char_count": len(cleaned)
                    })

        except pypdf.errors.PdfReadError as err:
            raise ValueError(f"Corrupted or unreadable PDF file '{path.name}': {err}")
        except Exception as err:
            raise RuntimeError(f"Unexpected error while reading PDF '{path.name}': {err}")

        if not segments:
            raise ValueError(
                f"No extractable text found in PDF '{path.name}'. "
                "The file may be image-only (scanned) or password-protected."
            )

        return segments
