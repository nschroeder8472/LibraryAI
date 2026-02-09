"""Query pipeline orchestration."""
from typing import Dict, Optional
import logging

from ..retrieval.retriever import Retriever
from ..retrieval.reranker import Reranker
from ..generation.model import LanguageModel
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

        self.llm = LanguageModel(
            model_name=config.generation.model_name,
            device=config.generation.device,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            do_sample=config.generation.do_sample,
            use_8bit=config.generation.use_8bit,
            use_4bit=config.generation.use_4bit
        )

        self.reranker = None
        if config.rerank.enabled:
            self.reranker = Reranker(
                model_name=config.rerank.model_name,
                device=config.rerank.device,
                top_n=config.rerank.top_n
            )

        self.templates = PromptTemplates()

    def query(self, query: str, query_type: str = "qa",
             reading_history: Optional[str] = None,
             scope: Optional[Dict] = None) -> Dict:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query
            query_type: Type of query (qa, recommendation, passage_location)
            reading_history: Optional reading history for recommendations
            scope: Optional scope filter for book/series-specific queries,
                   e.g. {"type": "book", "title": "..."} or
                        {"type": "series", "name": "..."}

        Returns:
            Dictionary with answer and retrieved contexts
        """
        logger.info(f"Processing query: {query[:50]}...")

        # Step 1: Retrieve relevant chunks (with optional scope filtering)
        logger.info("Retrieving relevant context...")
        retrieved_chunks = self.retriever.retrieve(query, scope=scope)

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in your library to answer this question.",
                "contexts": [],
                "query": query
            }

        # Step 2: Rerank for precision
        if self.reranker:
            logger.info("Reranking retrieved chunks...")
            retrieved_chunks = self.reranker.rerank(query, retrieved_chunks)

        # Step 3: Format context
        context = self.retriever.format_context(retrieved_chunks)

        # Step 4: Select prompt template
        if query_type == "recommendation":
            prompt = self.templates.recommendation_prompt(
                query, context, reading_history
            )
        elif query_type == "passage_location":
            prompt = self.templates.passage_location_prompt(query, context)
        else:  # default to qa
            prompt = self.templates.qa_prompt(query, context)

        # Step 5: Generate answer
        logger.info("Generating answer...")
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "contexts": retrieved_chunks,
            "query": query,
            "query_type": query_type
        }
