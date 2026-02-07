"""Prompt templates for RAG system."""
from typing import List, Dict, Optional

class PromptTemplates:
    """Collection of prompt templates."""

    @staticmethod
    def qa_prompt(query: str, context: str) -> str:
        """
        Question answering prompt.

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant answering questions about a personal book library.

Context from the library:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the provided context
- If the context contains the answer, provide it clearly and cite the book and chapter
- If the context doesn't contain enough information, say so honestly
- Be concise but complete

Answer:"""
        return prompt

    @staticmethod
    def recommendation_prompt(query: str, context: str,
                            reading_history: Optional[str] = None) -> str:
        """
        Book recommendation prompt.

        Args:
            query: User request for recommendations
            context: Retrieved context from similar books
            reading_history: Recent reading history

        Returns:
            Formatted prompt
        """
        history_section = ""
        if reading_history:
            history_section = f"\nRecent reading history:\n{reading_history}\n"

        prompt = f"""You are a helpful assistant recommending books from a personal library.

{history_section}
Relevant books from the library:
{context}

Request: {query}

Instructions:
- Recommend books from the library based on the request
- Consider the reading history if provided
- Explain why each recommendation fits the request
- Limit to 3-5 recommendations

Recommendations:"""
        return prompt

    @staticmethod
    def passage_location_prompt(query: str, context: str) -> str:
        """
        Find passage location prompt.

        Args:
            query: Query with passage to locate
            context: Retrieved context chunks

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant locating passages in a book library.

Context from the library:
{context}

Query: {query}

Instructions:
- Identify which book and chapter contains the passage or concept
- Provide specific location information (book title, author, chapter)
- If found in multiple places, list all occurrences
- If not found, say so clearly

Location:"""
        return prompt
