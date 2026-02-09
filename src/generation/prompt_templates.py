"""Prompt templates for RAG system."""
from typing import List, Dict, Optional


class PromptTemplates:
    """Collection of prompt templates."""

    @staticmethod
    def qa_prompt(query: str, context: str,
                  conversation_history: str = "") -> str:
        """
        Question answering prompt with optional conversation context.

        Args:
            query: User question
            context: Retrieved context
            conversation_history: Formatted prior conversation turns

        Returns:
            Formatted prompt
        """
        history_section = ""
        if conversation_history:
            history_section = f"""Previous conversation:
{conversation_history}

"""

        prompt = f"""You are a knowledgeable and friendly librarian assistant who helps users explore their personal ebook library. You are conversational but accurate.

{history_section}Relevant passages from the library:
{context}

Question: {query}

Instructions:
- Answer the question based on the provided passages. Do not use outside knowledge.
- Quote or closely paraphrase the relevant text when possible to support your answer.
- Always cite the specific book title and chapter where you found the information.
- If the user refers to something from the conversation history (like "that character" or "the book you mentioned"), resolve the reference using the history, but ground your answer in the library passages.
- If the passages do not contain enough information to answer confidently, say so warmly — for example, "I wasn't able to find details about that in the excerpts I have."
- Be conversational and natural. Start with the answer, then provide supporting evidence.

Answer:"""
        return prompt

    @staticmethod
    def recommendation_prompt(query: str, context: str,
                              reading_history: Optional[str] = None,
                              conversation_history: str = "") -> str:
        """Book recommendation prompt."""
        history_section = ""
        if reading_history:
            history_section += f"\nRecent reading history:\n{reading_history}\n"
        if conversation_history:
            history_section += f"\nPrevious conversation:\n{conversation_history}\n"

        prompt = f"""You are a friendly and knowledgeable librarian recommending books from a personal library.
{history_section}
Relevant books from the library:
{context}

Request: {query}

Instructions:
- Recommend books from the library based on the request
- Consider the reading history and conversation context if provided
- Explain why each recommendation fits the request
- Limit to 3-5 recommendations
- Be warm and conversational

Recommendations:"""
        return prompt

    @staticmethod
    def passage_location_prompt(query: str, context: str,
                                conversation_history: str = "") -> str:
        """Find passage location prompt."""
        history_section = ""
        if conversation_history:
            history_section = f"\nPrevious conversation:\n{conversation_history}\n"

        prompt = f"""You are a helpful librarian assistant locating passages in a book library.
{history_section}
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

    @staticmethod
    def chapter_summary_prompt(book_title: str, chapter_title: str,
                               chapter_text: str) -> str:
        """Summarize a single chapter."""
        return f"""Summarize the following chapter concisely. Focus on key events, character actions, and plot developments. Keep the summary to 2-4 paragraphs.

Book: "{book_title}"
Chapter: {chapter_title}

Text:
{chapter_text}

Summary:"""

    @staticmethod
    def book_summary_prompt(book_title: str, book_author: str,
                            chapter_summaries: str) -> str:
        """Generate a comprehensive book summary from chapter summaries."""
        return f"""You are summarizing a book based on chapter-by-chapter summaries. Write a cohesive, engaging summary that covers the main plot, key characters, and major themes. Keep it to 3-5 paragraphs.

Book: "{book_title}" by {book_author}

Chapter summaries:
{chapter_summaries}

Book summary:"""

    @staticmethod
    def series_summary_prompt(series_name: str, book_summaries: str) -> str:
        """Generate a series overview from book summaries."""
        return f"""You are summarizing a book series based on individual book summaries. Write an engaging overview that covers the overarching plot, main characters, and how the series evolves across books. Keep it to 3-5 paragraphs.

Series: "{series_name}"

Book summaries:
{book_summaries}

Series overview:"""

    @staticmethod
    def character_evolution_prompt(query: str, character: str,
                                   chronological_context: str) -> str:
        """Answer questions about character changes across books."""
        return f"""You are a knowledgeable librarian answering a question about a character's development across a book series. The passages below are sorted chronologically.

Character: {character}
Question: {query}

Chronological passages:
{chronological_context}

Instructions:
- Trace the character's development across the passages in chronological order.
- Note specific changes, events, or turning points.
- Cite the book and chapter for each key detail.
- If the passages don't contain enough information, say so.

Answer:"""

    @staticmethod
    def query_rewrite_prompt(conversation_history: str,
                             current_query: str) -> str:
        """Rewrite a follow-up query as a standalone question.

        Used when the user's query references prior conversation context
        (e.g. "What about her?" or "Tell me more about that").

        Args:
            conversation_history: Recent conversation turns
            current_query: The user's latest message

        Returns:
            Prompt that instructs the LLM to rewrite the query
        """
        return f"""Given the conversation history below, rewrite the user's latest message as a complete, standalone question that captures the full intent. Include specific names, titles, or details referenced in the conversation. If the latest message is already standalone, return it unchanged.

Conversation:
{conversation_history}

Latest message: {current_query}

Rewritten standalone question:"""
