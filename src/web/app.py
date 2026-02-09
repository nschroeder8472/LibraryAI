"""FastAPI web interface for LibraryAI."""
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..embeddings.vector_store import VectorStore
from ..pipeline.indexing import IndexingPipeline
from ..pipeline.query import QueryPipeline
from ..pipeline.summary import SummaryPipeline
from ..config import config
from ..utils.log_buffer import log_buffer
from .session import SessionManager

logger = logging.getLogger(__name__)

# Module-level state
pipeline: QueryPipeline | None = None
summary_pipeline: SummaryPipeline | None = None
session_manager = SessionManager()
indexing_status: dict = {"active": False, "message": "", "progress": "", "error": None}
summary_status: dict = {"active": False, "message": "", "error": None}
_indexing_lock = threading.Lock()
_summary_lock = threading.Lock()


class ScopeModel(BaseModel):
    type: str = ""  # "book", "series", or ""
    title: Optional[str] = None
    name: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    query_type: str = "qa"
    scope: Optional[ScopeModel] = None
    session_id: Optional[str] = None


class SummaryRequest(BaseModel):
    type: str  # "book" or "series"
    title: Optional[str] = None  # book title
    name: Optional[str] = None   # series name
    force: bool = False


class IndexRequest(BaseModel):
    library_dir: str | None = None


class _ProgressHandler(logging.Handler):
    """Captures 'Step N/5:' log messages from the indexing pipeline."""

    def emit(self, record):
        msg = self.format(record)
        if "Step " in msg and "/" in msg:
            indexing_status["progress"] = msg


def _run_indexing(library_dir: str):
    """Run indexing in a background thread."""
    global pipeline, summary_pipeline

    handler = _ProgressHandler()
    handler.setLevel(logging.INFO)
    indexing_logger = logging.getLogger("src.pipeline.indexing")
    indexing_logger.addHandler(handler)

    try:
        indexing_status["message"] = "Indexing in progress..."
        indexing_status["progress"] = "Starting..."
        indexing_status["error"] = None

        idx_pipeline = IndexingPipeline()
        vector_store = idx_pipeline.index_library(Path(library_dir))

        pipeline = QueryPipeline(vector_store)
        summary_pipeline = SummaryPipeline(vector_store, llm=pipeline.llm)
        indexing_status["message"] = "Indexing complete"
        indexing_status["progress"] = "Done"
        logger.info("Background indexing complete, query pipeline ready")
    except Exception as e:
        logger.error(f"Background indexing failed: {e}", exc_info=True)
        indexing_status["error"] = str(e)
        indexing_status["message"] = "Indexing failed"
    finally:
        indexing_status["active"] = False
        indexing_logger.removeHandler(handler)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load vector store and query pipeline on startup."""
    global pipeline, summary_pipeline
    try:
        logger.info("Loading vector store...")
        vector_store = VectorStore.load(config.data.vector_store_dir)
        stats = vector_store.get_stats()
        logger.info(f"Loaded index with {stats['total_vectors']} vectors")

        logger.info("Initializing query pipeline...")
        pipeline = QueryPipeline(vector_store)
        summary_pipeline = SummaryPipeline(vector_store, llm=pipeline.llm)
        logger.info("Query pipeline ready")
    except Exception as e:
        logger.warning(f"Could not load existing index: {e}")
        logger.info("Server starting without index. Build one from the web UI.")
        pipeline = None
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LibraryAI",
        description="RAG-based Q&A for your ebook library",
        lifespan=_lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(content=html_path.read_text())

    @app.get("/api/health")
    def health():
        has_index = pipeline is not None
        result = {
            "status": "ready" if has_index else "no_index",
            "has_index": has_index,
            "indexing": indexing_status["active"],
        }
        if has_index:
            try:
                result["index_stats"] = pipeline.retriever.vector_store.get_stats()
            except Exception as e:
                result["status"] = "error"
                result["message"] = str(e)
        return result

    @app.get("/api/library")
    def library_info():
        """Get library structure: series and ungrouped books."""
        if pipeline is None:
            return JSONResponse(
                content={"error": "No index available"},
                status_code=503,
            )
        try:
            return pipeline.retriever.vector_store.get_unique_series()
        except Exception as e:
            logger.error(f"Failed to get library info: {e}", exc_info=True)
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )

    # --- Session endpoints ---

    @app.post("/api/session/new")
    def new_session():
        """Create a new conversation session."""
        session = session_manager.create_session()
        return {"session_id": session.session_id}

    @app.post("/api/session/{session_id}/clear")
    def clear_session(session_id: str):
        """Clear conversation history for a session."""
        if session_manager.clear_session(session_id):
            return {"status": "cleared", "session_id": session_id}
        return JSONResponse(
            content={"error": "Session not found"},
            status_code=404,
        )

    # --- Indexing endpoints ---

    @app.post("/api/index")
    def start_indexing(request: IndexRequest):
        library_dir = request.library_dir or str(config.data.raw_dir)
        lib_path = Path(library_dir)

        if not lib_path.exists():
            return JSONResponse(
                content={"error": f"Directory does not exist: {library_dir}"},
                status_code=400,
            )

        epub_files = list(lib_path.glob("**/*.epub"))
        if not epub_files:
            return JSONResponse(
                content={"error": f"No EPUB files found in {library_dir}"},
                status_code=400,
            )

        with _indexing_lock:
            if indexing_status["active"]:
                return JSONResponse(
                    content={"error": "Indexing is already in progress"},
                    status_code=409,
                )
            indexing_status["active"] = True

        thread = threading.Thread(
            target=_run_indexing,
            args=(library_dir,),
            daemon=True,
        )
        thread.start()

        return JSONResponse(
            content={
                "message": f"Indexing started for {len(epub_files)} EPUB files",
                "epub_count": len(epub_files),
            },
            status_code=202,
        )

    @app.get("/api/index/status")
    def index_status():
        return indexing_status

    # --- Logs endpoint ---

    @app.get("/api/logs")
    def get_logs(level: str = None, limit: int = 200):
        """Return recent log entries from the in-memory buffer."""
        return log_buffer.get_logs(level=level, limit=limit)

    # --- Summary endpoints ---

    @app.post("/api/summary")
    def generate_summary(request: SummaryRequest):
        """Generate a book or series summary."""
        if pipeline is None:
            return JSONResponse(
                content={"error": "No index available"},
                status_code=503,
            )
        if summary_pipeline is None:
            return JSONResponse(
                content={"error": "Summary pipeline not initialized"},
                status_code=503,
            )

        with _summary_lock:
            if summary_status["active"]:
                return JSONResponse(
                    content={"error": "A summary is already being generated"},
                    status_code=409,
                )
            summary_status["active"] = True
            summary_status["error"] = None

        def _run():
            try:
                if request.type == "book" and request.title:
                    summary_status["message"] = f"Summarizing '{request.title}'..."
                    result = summary_pipeline.summarize_book(
                        request.title, force=request.force
                    )
                elif request.type == "series" and request.name:
                    summary_status["message"] = f"Summarizing series '{request.name}'..."
                    result = summary_pipeline.summarize_series(
                        request.name, force=request.force
                    )
                else:
                    summary_status["error"] = "Invalid summary request"
                    return

                summary_status["message"] = "done"
                summary_status["result"] = result
            except Exception as e:
                logger.error(f"Summary generation failed: {e}", exc_info=True)
                summary_status["error"] = str(e)
                summary_status["message"] = "Summary generation failed"
            finally:
                summary_status["active"] = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        label = request.title or request.name or ""
        return JSONResponse(
            content={"message": f"Summary generation started for '{label}'"},
            status_code=202,
        )

    @app.get("/api/summary/status")
    def get_summary_status():
        """Check summary generation progress."""
        return summary_status

    # --- Query endpoint ---

    @app.post("/api/query")
    def query(request: QueryRequest):
        if pipeline is None:
            if indexing_status["active"]:
                msg = "Indexing is in progress. Please wait until it completes."
            else:
                msg = "No index available. Please build one first using the setup panel."
            return JSONResponse(
                content={"error": msg},
                status_code=503,
            )
        try:
            # Get or create session
            session = session_manager.get_or_create_session(request.session_id)
            session_id = session.session_id

            # Update scope if provided
            scope = None
            if request.scope and request.scope.type:
                scope = {"type": request.scope.type}
                if request.scope.title:
                    scope["title"] = request.scope.title
                if request.scope.name:
                    scope["name"] = request.scope.name
                session.scope = scope

            # Get conversation history for the prompt
            conversation_history = session.format_history_for_prompt(
                max_turns=pipeline.max_conversation_turns
            )

            # Record user message in session
            session_manager.add_message(session_id, "user", request.query)

            # Run query with conversation context
            result = pipeline.query(
                request.query,
                query_type=request.query_type,
                scope=scope,
                conversation_history=conversation_history,
            )

            # Record assistant response in session
            session_manager.add_message(session_id, "assistant", result["answer"])

            # Strip non-serializable fields from contexts
            contexts = [
                {k: v for k, v in ctx.items()
                 if k not in ("embedding", "parent_text", "child_text", "parent_id")}
                for ctx in result.get("contexts", [])
            ]
            return {
                "answer": result["answer"],
                "contexts": contexts,
                "query": result["query"],
                "query_type": result.get("query_type", request.query_type),
                "session_id": session_id,
            }
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )

    return app
