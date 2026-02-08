"""FastAPI web interface for LibraryAI."""
import logging
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..embeddings.vector_store import VectorStore
from ..pipeline.indexing import IndexingPipeline
from ..pipeline.query import QueryPipeline
from ..config import config

logger = logging.getLogger(__name__)

# Module-level state
pipeline: QueryPipeline | None = None
indexing_status: dict = {"active": False, "message": "", "progress": "", "error": None}
_indexing_lock = threading.Lock()


class QueryRequest(BaseModel):
    query: str
    query_type: str = "qa"


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
    global pipeline

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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="LibraryAI", description="RAG-based Q&A for your ebook library")

    @app.on_event("startup")
    def startup():
        global pipeline
        try:
            logger.info("Loading vector store...")
            vector_store = VectorStore.load(config.data.vector_store_dir)
            stats = vector_store.get_stats()
            logger.info(f"Loaded index with {stats['total_vectors']} vectors")

            logger.info("Initializing query pipeline...")
            pipeline = QueryPipeline(vector_store)
            logger.info("Query pipeline ready")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            logger.info("Server starting without index. Build one from the web UI.")
            pipeline = None

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
            result = pipeline.query(request.query, query_type=request.query_type)
            return {
                "answer": result["answer"],
                "contexts": result.get("contexts", []),
                "query": result["query"],
                "query_type": result.get("query_type", request.query_type),
            }
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )

    return app
