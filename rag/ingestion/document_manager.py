import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class DocumentManager:
    """
    Manages document storage and metadata tracking for Ultron's knowledge base.

    Responsibilities:
    1. Persist uploaded document files locally under `rag/uploads/`.
    2. Maintain an index registry in `rag/index/documents.json`.
    3. Track metadata per document (id, filename, file_type, chunk count, upload status, timestamp).
    4. Provide methods to list, lookup, add, remove, and clear documents.
    """

    def __init__(
        self,
        uploads_dir: Optional[Path] = None,
        registry_path: Optional[Path] = None
    ):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.uploads_dir = uploads_dir or (project_root / "rag" / "uploads")
        self.registry_path = registry_path or (project_root / "rag" / "index" / "documents.json")

        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self._documents: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self):
        """Load document metadata from the JSON registry."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
            except Exception as err:
                print(f"[DocumentManager Warning] Failed to load registry '{self.registry_path}': {err}")
                self._documents = {}
        else:
            self._documents = {}

    def _save_registry(self):
        """Write document metadata to the JSON registry."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, indent=2, ensure_ascii=False)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return all tracked documents sorted by upload time (newest first)."""
        docs = list(self._documents.values())
        docs.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)
        return docs

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document record by its ID."""
        return self._documents.get(doc_id)

    def register_document(
        self,
        filename: str,
        stored_path: Path,
        file_type: str,
        number_of_chunks: int,
        file_size: int,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a newly uploaded and indexed document.

        :param filename: Original user filename (e.g. 'project_notes.txt').
        :param stored_path: Path where the file is stored under uploads.
        :param file_type: File extension/type (e.g. 'txt', 'md', 'pdf').
        :param number_of_chunks: Number of chunks extracted and indexed.
        :param file_size: Size of the file in bytes.
        :param doc_id: Optional custom document ID; generates a UUID if None.
        :return: Document record dictionary.
        """
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "document_id": doc_id,
            "filename": filename,
            "stored_path": str(stored_path.resolve()),
            "file_type": file_type.lstrip(".").lower(),
            "number_of_chunks": number_of_chunks,
            "file_size": file_size,
            "status": "indexed",
            "uploaded_at": now_str
        }

        self._documents[doc_id] = record
        self._save_registry()
        return record

    def remove_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Remove a document record and delete its physical file on disk.

        :param doc_id: ID of the document to remove.
        :return: The deleted record, or None if not found.
        """
        record = self._documents.pop(doc_id, None)
        if not record:
            return None

        # Delete physical file from uploads if present
        try:
            stored_path = Path(record.get("stored_path", ""))
            if stored_path.exists() and stored_path.is_file():
                stored_path.unlink()
        except Exception as err:
            print(f"[DocumentManager Warning] Failed to delete file for {doc_id}: {err}")

        self._save_registry()
        return record

    def clear(self):
        """Remove all tracked documents and their uploaded files."""
        for doc_id, record in list(self._documents.items()):
            try:
                stored_path = Path(record.get("stored_path", ""))
                if stored_path.exists() and stored_path.is_file():
                    stored_path.unlink()
            except Exception:
                pass

        self._documents = {}
        self._save_registry()
