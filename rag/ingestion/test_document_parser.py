import sys
import tempfile
from pathlib import Path
import pypdf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "rag" / "ingestion"))

from document_parser import DocumentParser


def test_txt_parsing():
    print("Testing TXT parsing...")
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("Ultron is a language model laboratory.\nBuilt from scratch using PyTorch.")
        temp_path = Path(f.name)

    try:
        segments = DocumentParser.parse(temp_path)
        assert len(segments) == 1
        assert "Ultron is a language model laboratory." in segments[0]["text"]
        assert segments[0]["page"] is None
        print("  [PASS] TXT parsed successfully.")
    finally:
        temp_path.unlink()


def test_md_parsing():
    print("Testing Markdown parsing...")
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
        f.write("# Ultron Knowledge Base\n\n- Fact 1: TinyStories dataset\n- Fact 2: BPE tokenizer")
        temp_path = Path(f.name)

    try:
        segments = DocumentParser.parse(temp_path)
        assert len(segments) == 1
        assert "Fact 1: TinyStories dataset" in segments[0]["text"]
        print("  [PASS] Markdown parsed successfully.")
    finally:
        temp_path.unlink()


def test_pdf_parsing():
    print("Testing PDF parsing with page numbers...")
    # Minimal valid 1-page PDF with text 'Ultron PDF Knowledge Test'
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (Ultron PDF Knowledge Test) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000338 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"407\n"
        b"%%EOF"
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb", delete=False) as f:
        f.write(pdf_bytes)
        temp_path = Path(f.name)

    try:
        segments = DocumentParser.parse(temp_path)
        assert len(segments) == 1
        assert segments[0]["page"] == 1
        assert "Ultron PDF Knowledge Test" in segments[0]["text"]
        print("  [PASS] Text extracted from PDF with page=1.")
    finally:
        temp_path.unlink()

    # Test empty/blank PDF detection
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb", delete=False) as f:
        writer.write(f)
        temp_blank = Path(f.name)

    try:
        try:
            DocumentParser.parse(temp_blank)
            assert False, "Should have raised ValueError on PDF with no extractable text"
        except ValueError as err:
            assert "No extractable text found" in str(err)
            print("  [PASS] Empty/blank PDF correctly detected and rejected.")
    finally:
        temp_blank.unlink()


def test_empty_file():
    print("Testing empty file handling...")
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        pass  # 0 bytes
        temp_path = Path(f.name)

    try:
        try:
            DocumentParser.parse(temp_path)
            assert False, "Should have raised ValueError on 0-byte file"
        except ValueError as err:
            assert "completely empty" in str(err)
            print("  [PASS] 0-byte file rejected cleanly.")
    finally:
        temp_path.unlink()


def test_unsupported_extension():
    print("Testing unsupported extension handling...")
    with tempfile.NamedTemporaryFile(suffix=".exe", mode="w", delete=False) as f:
        f.write("binary data")
        temp_path = Path(f.name)

    try:
        try:
            DocumentParser.parse(temp_path)
            assert False, "Should have raised ValueError on unsupported extension"
        except ValueError as err:
            assert "Unsupported file format" in str(err)
            print("  [PASS] Unsupported extension rejected cleanly.")
    finally:
        temp_path.unlink()


def main():
    print("=" * 60)
    print("RUNNING DOCUMENT PARSER TESTS")
    print("=" * 60)
    test_txt_parsing()
    test_md_parsing()
    test_pdf_parsing()
    test_empty_file()
    test_unsupported_extension()
    print("=" * 60)
    print("ALL DOCUMENT PARSER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
