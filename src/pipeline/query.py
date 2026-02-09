"""Query pipeline orchestration with conversation support."""
from typing import Dict, Generator, List, Optional, Tuple
import logging

from ..retrieval.retriever import Retriever
from ..retrieval.reranker import Reranker
from ..generation.model import create_language_model
from ..generation.prompt_templates import PromptTemplates
from ..embeddings.embedder import Embedder
from ..embeddings.vector_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Orchestrate the RAG query pipeline."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize query pipeline.

        Args:
            vector_store: Loaded vector store
        """
        self.embedder = Embedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )

        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=vector_store,
            top_k=config.retrieval.top_k,
            similarity_threshold=config.retrieval.similarity_threshold
        )

        # Use factory to create the right backend
        self.llm = create_language_model(
            backend=config.generation.backend,
            model_name=config.generation.model_name,
            device=config.generation.device,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            do_sample=config.generation.do_sample,
            use_8bit=config.generation.use_8bit,
            use_4bit=config.generation.use_4bit,
        )

        self.reranker = None
        if config.rerank.enabled:
            self.reranker = Reranker(
                model_name=config.rerank.model_name,
                device=config.rerank.device,
                top_n=config.rerank.top_n
            )

        self.templates = PromptTemplates()
        self.max_conversation_turns = config.generation.max_conversation_turns

    def query(self, query: str, query_type: str = "qa",
              reading_history: Optional[str] = None,
              scope: Optional[Dict] = None,
              conversation_history: str = "") -> Dict:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query
            query_type: Type of query (qa, recommendation, passage_location)
            reading_history: Optional reading history for recommendations
            scope: Optional scope filter for book/series-specific queries
            conversation_history: Formatted conversation history string

        Returns:
            Dictionary with answer and retrieved contexts
        """
        logger.info(f"Processing query: {query[:50]}...")

        _, retrieved_chunks = self._retrieve_and_rerank(
            query, query_type, scope, conversation_history
        )

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in your library to answer this question.",
                "contexts": [],
                "query": query,
            }

        prompt = self._build_prompt(
            query, query_type, retrieved_chunks,
            reading_history=reading_history,
            conversation_history=conversation_history,
        )

        logger.info("Generating answer...")
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "contexts": retrieved_chunks,
            "query": query,
            "query_type": query_type,
        }

    def _build_prompt(self, query: str, query_type: str,
                      retrieved_chunks: List[Dict],
                      reading_history: Optional[str] = None,
                      conversation_history: str = "") -> str:
        """Build the LLM prompt from query type and retrieved context."""
        context = self.retriever.format_context(retrieved_chunks)

        if query_type == "recommendation":
            return self.templates.recommendation_prompt(
                query, context, reading_history,
                conversation_history=conversation_history,
            )
        elif query_type == "passage_location":
            return self.templates.passage_location_prompt(
                query, context,
                conversation_history=conversation_history,
            )
        elif query_type == "character_evolution":
            return self.templates.character_evolution_prompt(
                query, query, context,
            )
        else:
            return self.templates.qa_prompt(
                query, context,
                conversation_history=conversation_history,
            )

    def _retrieve_and_rerank(self, query: str, query_type: str,
                             scope: Optional[Dict],
                             conversation_history: str
                             ) -> Tuple[str, List[Dict]]:
        """Run retrieval and reranking, return (retrieval_query, chunks).

        Returns an empty list if no chunks match.
        """
        retrieval_query = query
        if conversation_history:
            retrieval_query = self._rewrite_query(query, conversation_history)
            if retrieval_query != query:
                logger.info(f"Rewritten query: {retrieval_query[:80]}...")

        logger.info("Retrieving relevant context...")
        if query_type == "character_evolution":
            retrieved_chunks = self.retriever.retrieve_chronological(
                retrieval_query, scope=scope
            )
        else:
            retrieved_chunks = self.retriever.retrieve(
                retrieval_query, scope=scope
            )

        if self.reranker and retrieved_chunks:
            logger.info("Reranking retrieved chunks...")
            retrieved_chunks = self.reranker.rerank(
                retrieval_query, retrieved_chunks
            )

        return retrieval_query, retrieved_chunks

    def query_stream(self, query: str, query_type: str = "qa",
                     reading_history: Optional[str] = None,
                     scope: Optional[Dict] = None,
                     conversation_history: str = ""
                     ) -> Generator[Dict, None, None]:
        """Stream a query response token-by-token.

        Yields dicts with either:
          {"type": "contexts", "contexts": [...]} — once, at the start
          {"type": "token", "token": "..."} — for each generated token
          {"type": "done"} — when generation is complete

        If no chunks are retrieved, yields a single "no_results" message.
        """
        logger.info(f"Processing streaming query: {query[:50]}...")

        _, retrieved_chunks = self._retrieve_and_rerank(
            query, query_type, scope, conversation_history
        )

        if not retrieved_chunks:
            yield {
                "type": "no_results",
                "answer": "I couldn't find relevant information in your library to answer this question.",
            }
            return

        # Send contexts first so the UI can display sources
        yield {"type": "contexts", "contexts": retrieved_chunks}

        # Build prompt and stream generation
        prompt = self._build_prompt(
            query, query_type, retrieved_chunks,
            reading_history=reading_history,
            conversation_history=conversation_history,
        )

        logger.info("Streaming answer...")
        for token in self.llm.generate_stream(prompt):
            yield {"type": "token", "token": token}

        yield {"type": "done"}

    def _rewrite_query(self, query: str, conversation_history: str) -> str:
        """Rewrite a query using conversation context to make it standalone.

        Short or clearly standalone queries are returned as-is to save a
        round trip to the LLM.
        """
        # Heuristic: if the query is long enough and doesn't contain pronouns
        # or vague references, skip rewriting
        lower = query.lower()
        needs_rewrite = any(word in lower for word in [
            " he ", " she ", " they ", " it ", " that ", " this ",
            " him ", " her ", " them ", " those ",
            "tell me more", "what about", "how about",
            "the book", "the character", "the series",
            " his ", " its ",
        ])

        if not needs_rewrite:
            return query

        try:
            rewrite_prompt = self.templates.query_rewrite_prompt(
                conversation_history, query
            )
            rewritten = self.llm.generate(rewrite_prompt)

            # Basic validation: the rewritten query should not be empty or
            # wildly longer than the original (sanity check)
            if rewritten and len(rewritten) < len(query) * 5:
                return rewritten.strip()
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")

        return query
