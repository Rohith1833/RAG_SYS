class CharacterTokenizer:
    def __init__(self, text):
        # Find every unique character in the training text.
        self.characters = sorted(set(text))

        # Give every character an integer ID.
        self.char_to_id = {
            char: index
            for index, char in enumerate(self.characters)
        }

        # Reverse mapping: ID -> character.
        self.id_to_char = {
            index: char
            for char, index in self.char_to_id.items()
        }

        self.vocab_size = len(self.characters)

    def encode(self, text):
        """Convert text into a list of integer token IDs."""
        return [
            self.char_to_id[char]
            for char in text
        ]

    def decode(self, token_ids):
        """Convert token IDs back into text."""
        return "".join(
            self.id_to_char[token_id]
            for token_id in token_ids
        )