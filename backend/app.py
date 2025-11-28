from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

# Import our custom modules
from backend.database import vector_collection, conn
from backend.embeddings import get_embedding

app = FastAPI(title="Hybrid RAG System")

# Allow the frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.7  # 0.7 = Mostly Vector, 0.3 = Graph boost

# --- ROUTES ---

@app.get("/")
def home():
    return {"status": "System is running", "database": "ChromaDB + Kuzu"}

@app.post("/search")
def hybrid_search(req: SearchRequest):
    """
    Performs the Hybrid Search:
    1. Vector Search for similarity.
    2. Graph Traversal for context.
    3. Merges results.
    """
    # 1. Generate Embedding for the user's question
    query_vec = get_embedding(req.query)

    # 2. Vector Search (ChromaDB)
    vector_results = vector_collection.query(
        query_embeddings=[query_vec],
        n_results=req.top_k * 2 # Get more candidates for re-ranking
    )
    
    ids = vector_results['ids'][0]
    documents = vector_results['documents'][0]
    distances = vector_results['distances'][0]

    # 3. Hybrid Reranking
    final_results = []
    
    for i, doc_id in enumerate(ids):
        # Calculate a simple Vector Score (convert distance to similarity)
        # Chroma returns distance (lower is better), we want similarity (higher is better)
        vec_score = 1 / (1 + distances[i])
        
        # Graph Score: Check if this node is central or connected
        # (Simple Example: Count how many neighbors it has)
        graph_score = 0
        try:
            # Query Kuzu: "Count incoming connections to this node"
            query = f"MATCH (a)-[:NEXT]->(b) WHERE b.id = '{doc_id}' RETURN count(a)"
            result = conn.execute(query)
            while result.has_next():
                graph_score = result.get_next()[0]
        except:
            graph_score = 0
            
        # Normalize Graph Score (simple heuristic)
        graph_score_norm = min(graph_score * 0.1, 1.0)

        # Merge Scores
        final_score = (req.alpha * vec_score) + ((1 - req.alpha) * graph_score_norm)

        final_results.append({
            "id": doc_id,
            "text": documents[i],
            "score": final_score
        })

    # Sort by final score
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    return {"results": final_results[:req.top_k]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)