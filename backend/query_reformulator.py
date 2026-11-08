import re
from typing import List, Dict, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))


class QueryReformulator:
    """
    Resolves conversational follow-up questions into standalone semantic retrieval queries.
    
    Identifies ambiguous references (he, she, it, this, that, his, her, they, why?, when?)
    and uses recent conversation context to produce an explicit search query.
    Preserves the user's original query for the final generation turn.
    """

    PRONOUN_PATTERN = re.compile(
        r"\b(he|she|it|they|this|that|these|those|his|her|its|their|the author|the book|the model|why\?|when\?|how\?|who\?)\b",
        re.IGNORECASE
    )

    def __init__(self, backend=None):
        self.backend = backend

    def is_ambiguous(self, query: str) -> bool:
        """Heuristic check whether query contains pronouns or relative references."""
        q = query.strip()
        if len(q.split()) <= 4:
            # Very short follow-ups like 'When?', 'Why?', 'What else?'
            return True
        return bool(self.PRONOUN_PATTERN.search(q))

    def reformulate(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Reformulate an ambiguous follow-up query into a standalone query.
        
        :param query: Current user query.
        :param conversation_history: List of prior turns: [{'role': 'user'|'assistant', 'content': '...'}]
        :return: Standalone search query string.
        """
        clean_query = query.strip()
        if not conversation_history or not self.is_ambiguous(clean_query):
            return clean_query

        # If backend is available and online, ask it to reformulate
        if self.backend and hasattr(self.backend, "client"):
            try:
                # Take the last 2-4 turns for concise context
                recent = conversation_history[-4:]
                history_lines = []
                for turn in recent:
                    role = "User" if turn.get("role") == "user" else "Assistant"
                    history_lines.append(f"{role}: {turn.get('content', '').strip()}")
                history_text = "\n".join(history_lines)

                prompt = (
                    "You are a search query reformulation assistant.\n"
                    "Given the recent conversation history and a follow-up question, rewrite the follow-up "
                    "question into a concise, standalone semantic search query that replaces ambiguous pronouns "
                    "(e.g., he, she, it, that, his) with the explicit names, entities, or topics from the dialogue.\n\n"
                    "RULES:\n"
                    "1. Return ONLY the rewritten query in a single sentence.\n"
                    "2. Do NOT answer the question.\n"
                    "3. Do NOT add preamble or quotes.\n"
                    "4. If the question is already clear, return it as-is.\n\n"
                    f"RECENT CONVERSATION:\n{history_text}\n\n"
                    f"FOLLOW-UP QUESTION: {clean_query}\n\n"
                    "STANDALONE SEARCH QUERY:"
                )

                rewritten = self.backend.client.generate(
                    prompt=prompt,
                    temperature=0.1
                ).strip()

                # Clean up any quotes or markdown
                rewritten = re.sub(r'^["\']|["\']$', '', rewritten).strip()
                if rewritten and len(rewritten) > 3:
                    return rewritten
            except Exception:
                # Fallback on any connection error
                pass

        # Heuristic fallback: append key nouns from last assistant message
        last_turn = conversation_history[-1].get("content", "")
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", last_turn)
        if candidates:
            return f"{clean_query} ({candidates[0]})"

        return clean_query

