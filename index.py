from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Any
import openai
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import logging
import uuid
import json
import httpx
import requests
from github import Github
import mmh3
import datetime


load_dotenv()

# OpenAI Model Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Milvus Collection Configuration
MILVUS_COLLECTION_NAME = "github_dense_index"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI instance
app = FastAPI(title="RAG-Powered ChatGPT API with Zilliz Cloud", version="2.0.0")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    # Explicitly create an httpx client without proxies
    async_http_client = httpx.AsyncClient()
    # async_http_client = httpx.AsyncClient(proxies=None)
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=async_http_client)
else:
    client = None

# Global variables
collection = None
collection_github_sparse = None

def get_openai_client():
    """Get OpenAI client with error handling"""
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return client

# Initialize embedding model and collection
@app.on_event("startup")
async def startup_event():
    global collection, collection_github_sparse
    logger.info("Initializing RAG system...")
    
    # Connect to Zilliz Cloud
    logger.info("Connecting to Zilliz Cloud...")
    
    zilliz_cloud_uri = os.getenv("ZILLIZ_CLOUD_URI")
    zilliz_api_key = os.getenv("ZILLIZ_API_KEY")
    
    if not zilliz_cloud_uri or not zilliz_api_key:
        raise ValueError("ZILLIZ_CLOUD_URI and ZILLIZ_API_KEY must be set in environment variables")
    
    # Connect to Zilliz Cloud
    connections.connect(
        alias="default",
        uri=zilliz_cloud_uri,
        token=zilliz_api_key,
        secure=True
    )
    logger.info("Successfully connected to Zilliz Cloud!")
    
    # Initialize collection
    collection = setup_milvus_collection()

    # Initialize GitHub sparse collection (reserved for GitHub data only)
    try:
        collection_github_sparse = setup_github_sparse_collection()
        logger.info("GitHub sparse collection initialized")
    except Exception as e:
        logger.warning(f"Skipping GitHub sparse collection init: {e}")
    
    # Test OpenAI client initialization
    if client:
        logger.info("OpenAI client initialized successfully")
    else:
        logger.warning("OpenAI client not initialized - API key missing")
    
    logger.info("RAG system initialized successfully!")

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None

class DocumentUpload(BaseModel):
    title: str
    content: str

class DocumentResponse(BaseModel):
    id: str
    message: str

# Milvus setup
def setup_milvus_collection():
    collection_name = MILVUS_COLLECTION_NAME
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        logger.info(f"Connected to existing collection: {collection_name}")
        return collection
    
    # Create collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="repo", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="chunk", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)  # OpenAI text-embedding-3-small dimension
    ]
    
    schema = CollectionSchema(fields=fields, description="Knowledge base for RAG")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index for vector search
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    logger.info(f"Created new collection: {collection_name}")
    return collection

def setup_github_sparse_collection():
    """Create or load a separate sparse collection for GitHub-only data (BM25-like)."""
    collection_name = os.getenv("MILVUS_GITHUB_SPARSE_COLLECTION", "github_sparse_index")

    if utility.has_collection(collection_name):
        col = Collection(collection_name)
        logger.info(f"Connected to existing sparse collection: {collection_name}")
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
            FieldSchema(name="repo", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk", dtype=DataType.INT64),
            FieldSchema(name="sparse_emb", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, description="Sparse (BM25-like) index for GitHub files")
        col = Collection(name=collection_name, schema=schema)
        # Create sparse index
        index_params = {
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {}
        }
        col.create_index(field_name="sparse_emb", index_params=index_params)
        logger.info(f"Created new sparse collection: {collection_name}")
    return col

# Embedding functions
async def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI API"""
    try:
        # Get OpenAI client
        openai_client = get_openai_client()
        
        # Clean the text by removing newlines and extra whitespace
        cleaned_text = text.replace("\n", " ").strip()
        
        # Create embedding using the OpenAI client
        response = await openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=cleaned_text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

def chunk_text_with_overlap(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

async def store_document(title: str, content: str):
    # Get embedding
    chunks = chunk_text_with_overlap(content)
    for idx, chunk in enumerate(chunks):
        emb = await get_embedding(chunk)
        
        # Generate unique ID using mmh3
        doc_id = str(mmh3.hash128(title + f"_chunk_{idx}", signed=False))
        
        data = [
            [doc_id],  # id
            [""],      # repo (blank for non-GitHub)
            [title],   # title (without chunk number)
            [chunk],   # content
            [idx],     # chunk number
            [emb]      # embedding
        ]
        
        collection.insert(data)
    collection.load()
    return {"status": "Document stored successfully", "chunks": len(chunks)}

async def retrieve_relevant_docs(query: str, top_k: int = 3) -> List[dict]:
    """Retrieve relevant documents from Milvus"""
    query_embedding = await get_embedding(query)
    
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["repo", "title", "content", "chunk"]
    )
    
    docs = []
    for result in results[0]:
        docs.append({
            "repo": result.entity.get("repo"),
            "title": result.entity.get("title"),
            "content": result.entity.get("content"),
            "chunk": result.entity.get("chunk"),
            "score": result.score,
            "index_source": collection.name
        })
    
    return docs

async def retrieve_relevant_docs_sparse(query: str, top_k: int = 3) -> List[dict]:
    """Retrieve from GitHub sparse index using simple term-based query."""
    if collection_github_sparse is None:
        raise HTTPException(status_code=500, detail="Sparse collection not initialized")
    
    # For sparse, embed query as sparse vector (same tokenizer as ingest)
    query_sparse = _compute_sparse_embedding(query)
    if not query_sparse:
        return []
    
    search_params = {"metric_type": "IP", "params": {}}
    
    results = collection_github_sparse.search(
        data=[query_sparse],
        anns_field="sparse_emb",
        param=search_params,
        limit=top_k,
        output_fields=["repo", "path", "chunk"]
    )
    
    docs = []
    for result in results[0]:
        docs.append({
            "repo": result.entity.get("repo"),
            "path": result.entity.get("path"),
            "chunk": result.entity.get("chunk"),
            "score": result.score,
            "index_source": collection_github_sparse.name
        })
    return docs

async def rerank_documents(query: str, documents: List[dict]) -> List[dict]:
    """Re-rank documents using OpenAI LLM scoring."""
    openai_client = get_openai_client()
    scored_docs = []
    for doc in documents:
        prompt = f"Query: {query}\n\nDocument: {doc.get('content', doc.get('text', ''))[:500]}...\n\nRate the relevance of this document to the query on a scale of 0-10, where 10 is highly relevant. Respond with only the number."
        try:
            response = await openai_client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0.0
            )
            score = int(response.choices[0].message.content.strip())
        except:
            score = 0
        doc["rerank_score"] = score
        scored_docs.append(doc)
    return sorted(scored_docs, key=lambda x: x["rerank_score"], reverse=True)

async def generate_rag_response(query: str) -> tuple[str, List[Dict[str, str]]]:
    """Generate response using RAG (Retrieval-Augmented Generation)"""
    # Retrieve from both dense and sparse (GitHub-focused)
    dense_docs = await retrieve_relevant_docs(query, top_k=5)
    sparse_docs = await retrieve_relevant_docs_sparse(query, top_k=5)

    print(f"Dense index retrieval: For user prompt ->{query}<- loaded {len(dense_docs)} documents:\n")
    for doc in dense_docs:
        content = doc.get("content", doc.get("text", ""))
        title = doc.get("title", doc.get("path", "Unknown"))
        repo = doc.get("repo", "")
        chunk_num = doc.get("chunk", "")
        print(f"Repo: {repo}, Title: {title}, Chunk: {chunk_num}\nContent length: {len(content)}\n")
    print("\n")

    print(f"Sparse index retrieval: For user prompt ->{query}<- loaded {len(sparse_docs)} documents:\n")
    for doc in sparse_docs:
        # Sparse docs only have repo/path, no content stored
        path = doc.get("path", "Unknown")
        repo = doc.get("repo", "Unknown")
        chunk_num = doc.get("chunk", "")
        print(f"Repo: {repo}, Path: {path}, Chunk: {chunk_num}\nContent length: Not stored in sparse index\n")
    print("\n")

    # Combine and deduplicate by some key (e.g., title/path)
    combined_docs = {doc["title"] if "title" in doc else doc["path"]: doc for doc in dense_docs + sparse_docs}.values()
    
    # Re-rank
    reranked_docs = await rerank_documents(query, list(combined_docs))
    
    # Take top-5
    relevant_docs = reranked_docs[:5]
    
    # Build context from retrieved documents
    context = ""
    sources = []
    print(f"Final reranked RAG response: For user prompt ->{query}<- loaded {len(relevant_docs)} documents:\n")
    for doc in relevant_docs:
        content = doc.get("content", doc.get("text", ""))
        title = doc.get("title", doc.get("path", "Unknown"))
        repo = doc.get("repo", "")
        chunk_num = doc.get("chunk", "")
        context += f"Title: {title}\nContent: {content}\n\n"
        sources.append({"title": title, "index": doc["index_source"], "repo": repo, "chunk": str(chunk_num) if chunk_num is not None else ""})
        print(f"Repo: {repo}, Title: {title}, Chunk: {chunk_num} (from {doc["index_source"]})\nContent length: {len(content)}\n")
    print("\n")

    # Create RAG prompt with creativity control via instructions
    rag_prompt = f"""You are a precise, knowledgeable AI assistant. Use the following context to answer the user's question accurately and factually. Stay grounded in the provided information.

Context:
{context}

User Question: {query}

Instructions: Provide a clear, accurate response based on the context. If the context doesn't fully answer the question, clearly state what information is available and what is missing. Be direct and factual."""
    
    try:
        # Get OpenAI client and generate response
        openai_client = get_openai_client()
        response = await openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise, factual AI assistant. Provide accurate answers based strictly on the provided context. Avoid speculation or creative interpretation."},
                {"role": "user", "content": rag_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        return answer, sources
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# --- New: Ingest GitHub repo into Zilliz ---
class IngestGithubRequest(BaseModel):
    repo: str

def _fetch_github_files(repo_full_name: str) -> List[dict]:
    token = os.getenv('GITHUB_PAT')
    if not token:
        raise HTTPException(status_code=400, detail="GITHUB_PAT not configured on server")
    try:
        g = Github(token)
        repo = g.get_repo(repo_full_name)
        def walk(path: str = "") -> List[dict]:
            acc = []
            items = repo.get_contents(path)
            for item in items:
                if item.type == 'dir':
                    acc.extend(walk(item.path))
                else:
                    if item.path.endswith((".py", ".md", ".sql", ".yml")):
                        acc.append({"path": item.path, "content": str(item.decoded_content, 'utf-8', errors='ignore')})
            return acc
        return walk("")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub access error: {e}")

@app.post("/ingest-github")
async def ingest_github(req: IngestGithubRequest):
    # Use existing Milvus collection
    if collection is None:
        raise HTTPException(status_code=500, detail="Milvus collection not initialized")
    if collection_github_sparse is None:
        raise HTTPException(status_code=500, detail="GitHub sparse collection not initialized")
    print(f"[{datetime.datetime.now().isoformat()}] Starting ingestion for repo: {req.repo} into dense index: {collection.name} and sparse index: {collection_github_sparse.name}")
    # Upsert via Milvus SDK for this app
    files = _fetch_github_files(req.repo)
    count = 0
    for f in files:
        text = f.get("content", "")
        if not text:
            continue
        chunks = chunk_text_with_overlap(text)
        for chunk_idx, chunk in enumerate(chunks):
            try:
                # Dense insert
                emb = await get_embedding(chunk)
                chunk_id = str(mmh3.hash128(req.repo + '/' + f["path"] + f"_chunk_{chunk_idx}", signed=False))
                data = [
                    [chunk_id],                    # id
                    [req.repo],                    # repo
                    [f["path"]],                   # title (without chunk number)
                    [chunk],                       # content
                    [chunk_idx],                   # chunk number
                    [emb]                          # embedding
                ]
                collection.insert(data)
                
                # Sparse insert for the same chunk
                sparse = _compute_sparse_embedding(chunk)
                if sparse:
                    sparse_data = [
                        [chunk_id],
                        [req.repo],
                        [f["path"]],                   # path (without chunk number)
                        [chunk_idx],                   # chunk number
                        [sparse]
                    ]
                    collection_github_sparse.insert(sparse_data)
                
                count += 1
            except Exception as e:
                logger.warning(f"Skip {f.get('path')} chunk {chunk_idx}: {e}")
                continue
    collection.load()
    collection_github_sparse.load()
    print(f"[{datetime.datetime.now().isoformat()}] Ingestion completed for repo: {req.repo}. Files ingested: {count}")
    return {"files_ingested": count, "dense_collection": collection.name, "sparse_collection": collection_github_sparse.name}


# ---------------- Sparse GitHub Ingestion (BM25-like) ----------------
def _tokenize(text: str) -> List[str]:
    import re
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    # basic filtering
    return [t for t in tokens if len(t) >= 2]

def _compute_sparse_embedding(text: str) -> dict:
    """Compute a simple TF-weighted sparse embedding mapped by hashed term ids."""
    from collections import Counter
    terms = _tokenize(text)
    if not terms:
        return {}
    counts = Counter(terms)
    sparse = {}
    for term, tf in counts.items():
        idx = mmh3.hash(term, signed=False) % 1000000  # hash to a large dim space
        weight = 1.0 + (tf ** 0.5)  # sublinear tf
        sparse[idx] = weight
    # Sort by index
    return dict(sorted(sparse.items()))

def chunk_text(text: str, max_len: int = 20000) -> List[str]:
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

class IngestGithubSparseRequest(BaseModel):
    repo: str

@app.post("/ingest-github-sparse")
async def ingest_github_sparse(req: IngestGithubSparseRequest):
    if collection_github_sparse is None:
        raise HTTPException(status_code=500, detail="GitHub sparse collection not initialized")
    print(f"[{datetime.datetime.now().isoformat()}] Starting ingestion for repo: {req.repo} into index: {collection_github_sparse.name}")
    files = _fetch_github_files(req.repo)
    count = 0
    rows_id, rows_repo, rows_path, rows_chunk, rows_sparse = [], [], [], [], []
    for f in files:
        text = f.get("content", "")
        if not text:
            continue
        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            try:
                doc_id = str(mmh3.hash128(req.repo + '/' + f["path"] + f"_chunk_{chunk_idx}", signed=False))
                sparse = _compute_sparse_embedding(chunk)
                if sparse:
                    rows_id.append(doc_id)
                    rows_repo.append(req.repo)
                    rows_path.append(f["path"])
                    rows_chunk.append(chunk_idx)
                    rows_sparse.append(sparse)
                count += 1
            except Exception as e:
                logger.warning(f"Sparse skip {f.get('path')} chunk {chunk_idx}: {e}")
                continue

    if rows_id:
        collection_github_sparse.insert([
            rows_id,
            rows_repo,
            rows_path,
            rows_chunk,
            rows_sparse,
        ])
        collection_github_sparse.load()
    print(f"[{datetime.datetime.now().isoformat()}] Ingestion completed for repo: {req.repo}. Files ingested: {count}")
    return {"files_ingested": count, "collection": collection_github_sparse.name}

# API Endpoints

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("static/index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Chat Interface</h1><p>Please create static/index.html</p>")

# Original simple endpoints
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}!"}

@app.get("/health")
async def health_check():
    """Health check endpoint following RORO pattern"""
    # Descriptive variable names with auxiliary verbs
    has_openai_client = client is not None
    is_milvus_connected = utility.has_collection(MILVUS_COLLECTION_NAME)
    
    # Get collection counts and names (refresh from Zilliz source of truth)
    dense_count = 0
    sparse_count = 0
    dense_name = "N/A"
    sparse_name = "N/A"
    try:
        if collection:
            # Flush to ensure all operations are persisted
            collection.flush()
            # Get fresh count from Zilliz
            dense_count = collection.num_entities
            dense_name = collection.name
        if collection_github_sparse:
            # Flush to ensure all operations are persisted
            collection_github_sparse.flush()
            # Get fresh count from Zilliz
            sparse_count = collection_github_sparse.num_entities
            sparse_name = collection_github_sparse.name
    except Exception as e:
        logger.warning(f"Could not get collection counts: {e}")
    
    return {
        "webapp status": "healthy",
        "Zilliz Cloud Vector Index Status": is_milvus_connected,
        f"{dense_name} Count": dense_count,
        f"{sparse_name} Count": sparse_count,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "chat_model": OPENAI_CHAT_MODEL,
        "openai_status": "connected" if has_openai_client else "not configured"
    }

# RAG-powered chat endpoint
@app.post("/ask", response_model=ChatResponse)
async def chat_with_rag(message: ChatMessage):
    """Chat endpoint with RAG functionality"""
    query = message.message.strip()
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
    injection_keywords = ["ignore previous", "system prompt", "forget instructions"]
    if any(keyword in query.lower() for keyword in injection_keywords):
        raise HTTPException(status_code=400, detail="Potential prompt injection detected")
    try:
        # Generate RAG response
        response, sources = await generate_rag_response(message.message)
        
        return ChatResponse(response=response, sources=sources)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Document management endpoints
@app.post("/documents", response_model=DocumentResponse)
async def add_document(doc: DocumentUpload):
    """Add a document to the knowledge base"""
    try:
        doc_id = await store_document(doc.title, doc.content)
        return DocumentResponse(
            id=doc_id,
            message=f"Document '{doc.title}' added successfully to knowledge base"
        )
    except Exception as e:
        logger.error(f"Error storing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing document: {str(e)}")

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a text file to the knowledge base"""
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported")
    
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        doc_id = await store_document(file.filename, text_content)
        
        return DocumentResponse(
            id=doc_id,
            message=f"File '{file.filename}' uploaded successfully to knowledge base"
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/documents/search")
async def search_documents(query: str, limit: int = 5):
    """Search documents in the knowledge base"""
    try:
        docs = await retrieve_relevant_docs(query, top_k=limit)
        return {"query": query, "results": docs}
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

# Initialize with sample documents
@app.post("/init-sample-data")
async def initialize_sample_data():
    """Initialize the knowledge base with sample documents"""
    sample_docs = [
        {
            "title": "FastAPI Overview",
            "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints. It's built on top of Starlette and uses Pydantic for data validation."
        },
        {
            "title": "Vector Databases",
            "content": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They're essential for AI applications like semantic search, recommendation systems, and RAG (Retrieval-Augmented Generation)."
        },
        {
            "title": "Zilliz Cloud",
            "content": "Zilliz Cloud is a fully managed vector database service built on Milvus. It provides scalable similarity search and AI applications with enterprise-grade security, reliability, and performance optimization."
        }
    ]
    
    try:
        stored_docs = []
        for doc in sample_docs:
            doc_id = await store_document(doc["title"], doc["content"])
            stored_docs.append({"id": doc_id, "title": doc["title"]})
        
        return {
            "message": "Sample data initialized successfully",
            "documents": stored_docs
        }
    except Exception as e:
        logger.error(f"Error initializing sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing sample data: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)