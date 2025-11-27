
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3, json, math, os
from typing import List, Optional, Dict, Any
import numpy as np
import hashlib

DB_PATH = os.path.join(os.path.dirname(__file__), "vgdb.sqlite3")

# Try to load sentence-transformers, else use fallback
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_text(text):
        return MODEL.encode(text).tolist()
except Exception:
    MODEL = None
    def embed_text(text):
        # deterministic hash-based pseudo-embedding (fallback, reproducible)
        h = hashlib.sha256(text.encode('utf-8')).digest()
        vec = [((b % 128) - 64) / 64 for b in h[:64]]  # 64-d vector in [-1,1]
        return vec

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS nodes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        text TEXT,
        metadata TEXT,
        vector TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS edges(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source INTEGER,
        target INTEGER,
        type TEXT,
        weight REAL
    )""")
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="Vector+Graph Native DB (Prototype)")

class NodeIn(BaseModel):
    title: str
    text: str
    metadata: Optional[Dict[str,Any]] = {}
    embedding: Optional[List[float]] = None

class EdgeIn(BaseModel):
    source: int
    target: int
    type: Optional[str] = "related"
    weight: Optional[float] = 1.0

@app.post("/nodes")
def create_node(node: NodeIn):
    vec = node.embedding if node.embedding else embed_text(node.text)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO nodes(title,text,metadata,vector) VALUES (?,?,?,?)",
                (node.title, node.text, json.dumps(node.metadata), json.dumps(vec)))
    nid = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": nid, "title": node.title}

@app.get("/nodes/{node_id}")
def get_node(node_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM nodes WHERE id=?", (node_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"id": row["id"], "title": row["title"], "text": row["text"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "vector": json.loads(row["vector"] or "[]")}

@app.put("/nodes/{node_id}")
def update_node(node_id: int, node: NodeIn):
    vec = node.embedding if node.embedding else embed_text(node.text)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE nodes SET title=?, text=?, metadata=?, vector=? WHERE id=?",
                (node.title, node.text, json.dumps(node.metadata), json.dumps(vec), node_id))
    conn.commit()
    conn.close()
    return {"id": node_id, "updated": True}

@app.delete("/nodes/{node_id}")
def delete_node(node_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM edges WHERE source=? OR target=?", (node_id,node_id))
    cur.execute("DELETE FROM nodes WHERE id=?", (node_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}

@app.post("/edges")
def create_edge(edge: EdgeIn):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO edges(source,target,type,weight) VALUES (?,?,?,?)",
                (edge.source, edge.target, edge.type, edge.weight))
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": eid}

@app.get("/edges/{edge_id}")
def get_edge(edge_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM edges WHERE id=?", (edge_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Edge not found")
    return dict(row)

# Utilities for vector math
def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def all_nodes():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM nodes")
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({"id": r["id"], "title": r["title"], "text": r["text"],
                        "metadata": json.loads(r["metadata"] or "{}"),
                        "vector": json.loads(r["vector"] or "[]")})
    return results

@app.post("/search/vector")
def vector_search(body: dict):
    query_text = body.get("query_text", "")
    top_k = int(body.get("top_k", 5))
    qvec = embed_text(query_text)
    nodes = all_nodes()
    scored = []
    for n in nodes:
        sim = cosine_sim(qvec, n["vector"])
        scored.append((sim, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "node": node} for s,node in scored[:top_k]]

# Graph traversal (BFS up to depth)
@app.get("/search/graph")
def graph_traversal(start_id: int, depth: int = 1):
    conn = get_conn()
    cur = conn.cursor()
    # build adjacency
    cur.execute("SELECT source, target FROM edges")
    rows = cur.fetchall()
    adj = {}
    for r in rows:
        adj.setdefault(r["source"], []).append(r["target"])
    visited = set()
    from collections import deque
    q = deque()
    q.append((start_id, 0))
    visited.add(start_id)
    results = []
    while q:
        node, d = q.popleft()
        results.append({"id": node, "distance": d})
        if d < depth:
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d+1))
    # fetch node details
    out = []
    cur2 = conn.cursor()
    for r in results:
        cur2.execute("SELECT * FROM nodes WHERE id=?", (r["id"],))
        row = cur2.fetchone()
        if row:
            out.append({"node": {"id": row["id"], "title": row["title"], "text": row["text"]}, "distance": r["distance"]})
    conn.close()
    return out

@app.post("/search/hybrid")
def hybrid_search(body: dict):
    query_text = body.get("query_text", "")
    top_k = int(body.get("top_k",5))
    vector_weight = float(body.get("vector_weight", 0.6))
    graph_weight = float(body.get("graph_weight", 0.4))
    # embed query
    qvec = embed_text(query_text)
    # compute vector sims
    nodes = all_nodes()
    node_scores = {}
    for n in nodes:
        sim = cosine_sim(qvec, n["vector"])
        node_scores[n["id"]] = {"vector_sim": sim, "node": n, "graph_distance": None}
    # simple graph distance from top vector matches as start points
    # pick top 1 vector match as starting node for BFS
    top_vector_sorted = sorted(node_scores.items(), key=lambda x: x[1]["vector_sim"], reverse=True)
    start_nodes = [top_vector_sorted[0][0]] if top_vector_sorted else []
    # BFS to compute distances up to depth 3
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT source, target FROM edges")
    rows = cur.fetchall()
    adj = {}
    for r in rows:
        adj.setdefault(r["source"], []).append(r["target"])
    from collections import deque
    distances = {}
    for s in start_nodes:
        q = deque()
        q.append((s,0))
        seen = set([s])
        while q:
            node,d = q.popleft()
            distances.setdefault(node, d if node not in distances else min(distances[node], d))
            if d < 3:
                for nb in adj.get(node,[]):
                    if nb not in seen:
                        seen.add(nb)
                        q.append((nb, d+1))
    # combine scores
    final = []
    for nid, info in node_scores.items():
        vec_sim = info["vector_sim"]
        gd = distances.get(nid, None)
        graph_score = 1.0/(gd+1) if gd is not None else 0.0
        final_score = vector_weight * vec_sim + graph_weight * graph_score
        final.append((final_score, {"id": nid, "title": info["node"]["title"], "text": info["node"]["text"], "vector_sim": vec_sim, "graph_distance": gd}))
    final.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "node": n} for s,n in final[:top_k]]
