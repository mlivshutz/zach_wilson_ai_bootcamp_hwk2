from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import logging
import uuid
import json
import httpx
import mmh3

# Load environment variables
load_dotenv()

# OpenAI Model Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Milvus Collection Configuration
MILVUS_COLLECTION_NAME = "ansh_lamba_databricks_videos"

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

def get_openai_client():
    """Get OpenAI client with error handling"""
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return client

# Initialize embedding model and collection
@app.on_event("startup")
async def startup_event():
    global collection
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
    sources: Optional[List[str]] = None

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
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
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

async def store_document(title: str, content: str) -> str:
    """Store document in Milvus vector database"""
    # Generate a content-based ID using Murmur3 hash
    doc_id = str(mmh3.hash128(content, signed=False))
    
    embedding = await get_embedding(content)
    
    # Insert data
    data = [
        [doc_id],  # id
        [title],   # title
        [content], # content
        [embedding] # embedding
    ]
    
    collection.insert(data)
    collection.load()  # Load collection to memory for search
    
    logger.info(f"Stored document: {title} with ID: {doc_id}")
    return doc_id

async def retrieve_relevant_docs(query: str, top_k: int = 3) -> List[dict]:
    """Retrieve relevant documents from Milvus"""
    query_embedding = await get_embedding(query)
    
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "content"]
    )
    
    docs = []
    for result in results[0]:
        docs.append({
            "title": result.entity.get("title"),
            "content": result.entity.get("content"),
            "score": result.score
        })
    
    return docs

async def generate_rag_response(query: str) -> tuple[str, List[str]]:
    """Generate response using RAG (Retrieval-Augmented Generation)"""
    # Retrieve relevant documents
    relevant_docs = await retrieve_relevant_docs(query, top_k=3)
    
    # Build context from retrieved documents
    context = ""
    sources = []
    print(f"For user prompt ->{query}<- loaded {len(relevant_docs)} documents:\n")
    for doc in relevant_docs:
        context += f"Title: {doc['title']}\nContent: {doc['content']}\n\n"
        sources.append(doc['title'])
        print(f"Title: {doc['title']}\nContent length: {len(doc['content'])}\n")
    print("\n")

    # Create RAG prompt
    rag_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so and provide a general response.

Context:
{context}

User Question: {query}

Please provide a helpful and accurate response based on the context provided:"""
    
    try:
        # Get OpenAI client and generate response
        openai_client = get_openai_client()
        response = await openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
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
    
    return {
        "webapp status": "healthy",
        "Vector Index": "Zilliz Cloud",
        "Vector Index database": MILVUS_COLLECTION_NAME,
        "Vector Index status": is_milvus_connected,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "chat_model": OPENAI_CHAT_MODEL,
        "openai_status": "connected" if has_openai_client else "not configured"
    }

# RAG-powered chat endpoint
@app.post("/ask", response_model=ChatResponse)
async def chat_with_rag(message: ChatMessage):
    """Chat endpoint with RAG functionality"""
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