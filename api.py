from fastapi import FastAPI, HTTPException  # FastAPI web framework and HTTP error handling
from pydantic import BaseModel  # data validation for request/response models
from typing import List, Optional  # type hints for response schemas
import logging  # logging setup for server events
from retriever import NCORetriever  # vector search component using FAISS embeddings
from rag import NCORAG  # retrieval-augmented generation component

logging.basicConfig(level=logging.INFO)  # log INFO and higher messages
logger = logging.getLogger(__name__)  # module logger

app = FastAPI(
    title="NCO 2015 RAG System",
    description="AI Career Advisor using NCO 2015 occupational data",
    version="1.0.0"
)  # web app metadata for automatic docs

try:
    # Build the retrieval component and attach it to the RAG system
    retriever = NCORetriever()
    rag = NCORAG(retriever)
    logger.info("✅ RAG system initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG system: {e}")
    retriever = None
    rag = None

class Query(BaseModel):
    query: str
    top_k: int = 5
    
    class Config:
        schema_extra = {
            "example": {
                "query": "I enjoy working with machines and designing mechanical systems",
                "top_k": 5
            }
        }

class OccupationSource(BaseModel):
    title: str
    nco_2015: str
    description: str
    similarity_score: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    generated_answer: str
    sources: List[OccupationSource]
    status: str = "success"

class ErrorResponse(BaseModel):
    query: str
    error: str
    status: str = "error"

@app.get("/", response_model=dict)
def health():
    """Health check endpoint"""
    status = {
        "status": "ok",
        "rag_initialized": rag is not None,
        "retriever_initialized": retriever is not None
    }
    if rag is None:
        status["status"] = "degraded"
        status["message"] = "RAG system not initialized"
    return status

@app.post("/search", response_model=SearchResponse, responses={500: {"model": ErrorResponse}})
async def search_jobs(q: Query):
    """Search for occupations matching the query."""
    if rag is None:
        raise HTTPException(
            status_code=500,
            detail="RAG system not initialized. Check server logs."
        )

    if not q.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        logger.info(f"Processing query: {q.query[:50]}...")
        result = rag.generate_answer(q.query, q.top_k)  # generate RAG answer and source list

        sources = [
            OccupationSource(
                title=r.get('title', 'Unknown'),
                nco_2015=r.get('nco_2015', 'N/A'),
                description=r.get('description', 'No description available'),
                similarity_score=r.get('score', None)
            )
            for r in result["sources"]
        ]

        return SearchResponse(
            query=q.query,
            generated_answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))