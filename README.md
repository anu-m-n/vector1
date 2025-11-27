
Vector + Graph Native Database — Minimal Hackathon Prototype
===========================================================

What this project contains
--------------------------
- backend/
  - main.py        -> FastAPI backend with CRUD for nodes/edges, vector/graph/hybrid search
  - populate.py    -> script to populate the DB with sample dataset
  - requirements.txt
  - sample_data.json -> sample nodes and edges

- frontend/
  - index.html    -> single-file UI to test searches and CRUD

How to run
----------
1. Create a Python venv and install requirements:
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r backend/requirements.txt

2. Run the backend:
   uvicorn backend.main:app --reload --port 8000

3. Open the frontend:
   open frontend/index.html in your browser (or use Live Server extension)

Notes
-----
- The backend tries to import `sentence_transformers`. If it's available it will use a small model to create real embeddings.
  If not available, it falls back to a deterministic mocked embedding generator (based on hashing) so results are reproducible.

- Vectors are stored in SQLite as JSON arrays. Edges and nodes use SQLite tables.

- Hybrid scoring = vector_weight * cosine_similarity + graph_weight * (1/(distance+1))

Project structure and key design decisions are documented in the source files.
