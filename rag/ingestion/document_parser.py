import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
import pypdf


class DocumentParser:
    """
    A lightweight, multi-format document parser for the Ultron RAG system.
    
    Responsibilities:
    1. Validate document file types (.txt, .md, .pdf, .docx).
    2. Extract raw text from text-based documents (.txt, .md).
    3. Extract text page-by-page from PDF documents (.pdf) using pypdf to preserve page numbers.
    4. Extract text from Word documents (.docx) using built-in XML parsing.
    5. Handle empty, scanned, encrypted, or corrupted files with clear user error messages.
    6. Provide structured document segments ready for text chunking.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

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
            supported = ", ".join(sorted(cls.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file format '{ext}'. Ultron currently supports: {supported}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"The uploaded document '{path.name}' is completely empty (0 bytes).")

        if ext in {".txt", ".md"}:
            return cls._parse_text_file(path)
        elif ext == ".pdf":
            return cls._parse_pdf_file(path)
        elif ext == ".docx":
            return cls._parse_docx_file(path)
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

            if reader.is_encrypted:
                try:
                    # Attempt decrypt with empty password for trivial read-only protections
                    reader.decrypt("")
                except Exception:
                    raise ValueError(
                        f"PDF '{path.name}' is password-protected or encrypted. "
                        "Please provide an unencrypted PDF."
                    )

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
        except ValueError:
            raise
        except Exception as err:
            raise RuntimeError(f"Unexpected error while reading PDF '{path.name}': {err}")

        if not segments:
            raise ValueError(
                f"No extractable text found in PDF '{path.name}'. "
                "The file appears to be scanned or image-based. Optical Character Recognition (OCR) "
                "is required to extract text from scanned documents, but OCR is currently unavailable."
            )

        return segments

    @classmethod
    def _parse_docx_file(cls, path: Path) -> List[Dict[str, Any]]:
        """Extract text from Word .docx files using zipfile and XML parsing."""
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                if "word/document.xml" not in zf.namelist():
                    raise ValueError(
                        f"Corrupted or unsupported Word document '{path.name}': missing word/document.xml"
                    )
                xml_data = zf.read("word/document.xml")

            tree = ET.fromstring(xml_data)
            # OpenXML namespace
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

            paragraphs = []
            for p in tree.iterfind(".//w:p", ns):
                texts = [t.text for t in p.iterfind(".//w:t", ns) if t.text]
                if texts:
                    paragraph_text = "".join(texts).strip()
                    if paragraph_text:
                        paragraphs.append(paragraph_text)

            full_text = "\n\n".join(paragraphs).strip()

            if not full_text:
                raise ValueError(f"Word document '{path.name}' contains no readable text content.")

            return [{
                "page": None,
                "text": full_text,
                "char_count": len(full_text)
            }]

        except zipfile.BadZipFile:
            raise ValueError(f"Corrupted Word file '{path.name}': Not a valid ZIP/DOCX archive.")
        except ET.ParseError as pe:
            raise ValueError(f"Corrupted XML structure in Word document '{path.name}': {pe}")
        except ValueError:
            raise
        except Exception as err:
            raise RuntimeError(f"Unexpected error while parsing DOCX '{path.name}': {err}")
