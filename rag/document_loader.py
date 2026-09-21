from pathlib import Path


class DocumentLoader:
    """
    A simple, beginner-friendly text document loader.
    Reads local text files using UTF-8 encoding.
    """

    @staticmethod
    def load_text(file_path: str | Path) -> str:
        """
        Load and return the contents of a text file.

        :param file_path: Path to the text file.
        :return: Entire text content as a string.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return content
