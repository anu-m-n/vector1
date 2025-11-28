import os
import chromadb
import kuzu

# --- CONFIGURATION ---
# This trick finds the root folder no matter where you run the code from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
GRAPH_DB_PATH = os.path.join(DATA_DIR, "kuzu_db")

# --- AUTO-CREATE FOLDERS ---
if not os.path.exists(DATA_DIR):
    print(f"Creating data folder at: {DATA_DIR}")
    os.makedirs(DATA_DIR)

# --- 1. SETUP VECTOR DB (ChromaDB) ---
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
vector_collection = chroma_client.get_or_create_collection(name="knowledge_base")

# --- 2. SETUP GRAPH DB (Kuzu) ---
db = kuzu.Database(GRAPH_DB_PATH)
conn = kuzu.Connection(db)

def init_db():
    """Creates the graph tables if they don't exist."""
    try:
        conn.execute("CREATE NODE TABLE Entity(id STRING, text STRING, PRIMARY KEY (id))")
        conn.execute("CREATE REL TABLE RELATED(FROM Entity TO Entity)")
        print("Graph Schema Initialized.")
    except RuntimeError:
        pass # Tables already exist, which is fine

init_db()