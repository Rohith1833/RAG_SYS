import time
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag" / "ingestion"))

from ultron_backend import UltronBackend
from document_parser import DocumentParser


class DocumentSummarizer:
    """
    Hierarchical long-document summarization engine for Ultron.
    
    Implements a map-reduce summarization pipeline:
    1. Extracts full document content page-by-page or section-by-section.
    2. Breaks large documents into coherent sections (~2,000 - 3,500 chars).
    3. Intermediate Map step: Summarizes each individual section.
    4. Synthesis Reduce step: Combines section summaries into a final coherent summary.
    5. Supports 'short', 'detailed', and 'key_points' styles.
    """

    def __init__(self, backend: Optional[UltronBackend] = None):
        self.backend = backend if backend is not None else UltronBackend(model="llama3:latest")

    def summarize_document(
        self,
        file_path: Path | str,
        style: str = "detailed",
        max_section_chars: int = 3000
    ) -> Dict[str, Any]:
        """
        Produce a complete document summary.

        :param file_path: Path to the target document.
        :param style: 'short', 'detailed', or 'key_points'.
        :param max_section_chars: Max characters per section in hierarchical map phase.
        :return: Dict containing summary text, style, section count, metrics.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found at: {path}")

        t_start = time.perf_counter()

        # Step 1: Parse entire document text with structural segments
        segments = DocumentParser.parse(path)
        if not segments:
            raise ValueError(f"No readable content could be extracted from '{path.name}'.")

        # Aggregate total document text
        total_chars = sum(s.get("char_count", len(s.get("text", ""))) for s in segments)

        # Step 2: Determine if single-pass or multi-section hierarchical reduction
        style_clean = style.lower().strip()
        if style_clean not in {"short", "detailed", "key_points"}:
            style_clean = "detailed"

        if total_chars <= max_section_chars:
            # Single-pass direct summarization
            combined_text = "\n\n".join(
                (f"[Page {s['page']}]\n" if s.get("page") else "") + s["text"]
                for s in segments
            )
            final_summary = self._summarize_single_pass(
                text=combined_text,
                doc_name=path.name,
                style=style_clean
            )
            sections_count = 1
        else:
            # Hierarchical Map-Reduce summarization
            sections = self._build_sections(segments, max_chars=max_section_chars)
            sections_count = len(sections)

            section_summaries = []
            for idx, sec in enumerate(sections, start=1):
                sec_summary = self._summarize_section(
                    section_text=sec["text"],
                    section_num=idx,
                    total_sections=len(sections),
                    page_range=sec.get("page_range")
                )
                section_summaries.append(f"Section {idx} Summary:\n{sec_summary}")

            # Synthesize all section summaries into final document summary
            combined_summaries = "\n\n---\n\n".join(section_summaries)
            final_summary = self._synthesize_final_summary(
                combined_summaries=combined_summaries,
                doc_name=path.name,
                style=style_clean,
                total_sections=len(sections)
            )

        elapsed_sec = time.perf_counter() - t_start

        return {
            "summary": final_summary,
            "document_name": path.name,
            "style": style_clean,
            "total_characters": total_chars,
            "sections_processed": sections_count,
            "elapsed_seconds": round(elapsed_sec, 2)
        }

    def _build_sections(
        self,
        segments: List[Dict[str, Any]],
        max_chars: int
    ) -> List[Dict[str, Any]]:
        """Cluster segments into section blocks of ~max_chars."""
        sections = []
        current_text = []
        current_pages = []
        current_len = 0

        for seg in segments:
            text = seg.get("text", "").strip()
            page = seg.get("page")

            if current_len + len(text) > max_chars and current_text:
                sections.append({
                    "text": "\n\n".join(current_text),
                    "page_range": f"Pages {min(current_pages)}-{max(current_pages)}" if current_pages and current_pages[0] is not None else None
                })
                current_text = []
                current_pages = []
                current_len = 0

            current_text.append((f"[Page {page}]\n" if page else "") + text)
            if page is not None:
                current_pages.append(page)
            current_len += len(text)

        if current_text:
            sections.append({
                "text": "\n\n".join(current_text),
                "page_range": f"Pages {min(current_pages)}-{max(current_pages)}" if current_pages and current_pages[0] is not None else None
            })

        return sections

    def _summarize_single_pass(self, text: str, doc_name: str, style: str) -> str:
        """Summarize text within model context limit in one prompt."""
        instructions = self._get_style_instructions(style)
        prompt = (
            f"You are Ultron, an expert AI document summarizer.\n\n"
            f"DOCUMENT TITLE: {doc_name}\n\n"
            f"STYLE REQUIREMENTS:\n{instructions}\n\n"
            f"DOCUMENT CONTENT:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"FINAL SUMMARY:"
        )
        return self.backend.respond(prompt)

    def _summarize_section(
        self,
        section_text: str,
        section_num: int,
        total_sections: int,
        page_range: Optional[str]
    ) -> str:
        """Summarize a single section of a large document."""
        page_info = f" ({page_range})" if page_range else ""
        prompt = (
            f"You are Ultron, an AI assistant analyzing a large document in parts.\n"
            f"Summarize Section {section_num} of {total_sections}{page_info}.\n"
            f"Extract all critical facts, key arguments, names, figures, and concepts. "
            f"Be factual and concise.\n\n"
            f"SECTION TEXT:\n\"\"\"\n{section_text}\n\"\"\"\n\n"
            f"SECTION {section_num} SUMMARY:"
        )
        return self.backend.respond(prompt)

    def _synthesize_final_summary(
        self,
        combined_summaries: str,
        doc_name: str,
        style: str,
        total_sections: int
    ) -> str:
        """Combine section summaries into a cohesive, structured final document summary."""
        instructions = self._get_style_instructions(style)
        prompt = (
            f"You are Ultron, synthesizing a comprehensive summary for the document: '{doc_name}'.\n"
            f"Below are the extracted summaries from all {total_sections} sections of the document.\n\n"
            f"SYNTHESIS INSTRUCTIONS:\n"
            f"1. Produce a unified, fluent, and comprehensive summary covering the whole document.\n"
            f"2. Eliminate redundancy across sections.\n"
            f"3. Organize logically with clear headings or structured paragraphs.\n"
            f"4. Follow the requested style:\n{instructions}\n\n"
            f"SECTION EXTRACTS:\n{combined_summaries}\n\n"
            f"FINAL SYNTHESIZED DOCUMENT SUMMARY:"
        )
        return self.backend.respond(prompt)

    @staticmethod
    def _get_style_instructions(style: str) -> str:
        if style == "short":
            return (
                "- Provide an Executive Summary in 2-3 concise paragraphs.\n"
                "- State the main theme, critical arguments, and high-level conclusion."
            )
        elif style == "key_points":
            return (
                "- Present the document as a structured list of Key Points and Takeaways.\n"
                "- Include important facts, names, metrics, and core concepts.\n"
                "- Conclude with a brief 2-sentence summary of the overall conclusion."
            )
        else:  # detailed
            return (
                "- Provide a Detailed, Comprehensive Summary.\n"
                "- Structure into: 1. Overview & Objective, 2. Key Sections & Findings, "
                "3. Crucial Details & Technical Concepts, 4. Conclusion & Implications."
            )

