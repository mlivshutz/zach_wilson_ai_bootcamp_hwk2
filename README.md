# Zach Wilson AI Bootcamp - Homework 1

## 🤖 RAG-Powered ChatGPT Application with Zilliz Cloud

This is an advanced AI-powered chat application that combines **ChatGPT** with **RAG (Retrieval-Augmented Generation)** using **Zilliz Cloud** managed vector database for intelligent document retrieval and contextual responses.

## ✨ Features

### 🚀 Core Features
- **RAG-Powered Conversations** - AI responses enhanced with relevant document context
- **Zilliz Cloud Integration** - High-performance managed vector database with auto-scaling
- **Real-time Chat Interface** - Beautiful, responsive web UI with typing indicators
- **Document Management** - Upload files, add text documents, and manage knowledge base
- **Source Attribution** - See which documents were used to generate responses

### 🎯 AI & Vector Search
- **Semantic Search** - Find relevant documents using vector embeddings
- **Context-Aware Responses** - ChatGPT uses retrieved documents as context
- **OpenAI Embeddings** - Uses OpenAI's text-embedding-3-small model for high-quality vectorization
- **Cosine Similarity** - Efficient similarity matching for relevant content retrieval

### 📁 Document Support
- **File Upload** - Support for `.txt` and `.md` files
- **Manual Text Entry** - Add documents directly through the web interface
- **Sample Data** - Pre-loaded examples to get started quickly
- **Document Search** - Find and explore your knowledge base

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │────│  FastAPI Server │────│  Zilliz Cloud   │
│                 │    │                 │    │                 │
│ • Chat UI       │    │ • RAG Pipeline  │    │ • Vector Store  │
│ • File Upload   │    │ • Embeddings    │    │ • Similarity    │
│ • Document Mgmt │    │ • OpenAI API    │    │   Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### RAG Pipeline Flow:
1. **Document Ingestion** → Vectorize and store in Zilliz Cloud
2. **User Query** → Generate query embedding
3. **Retrieval** → Find similar documents using cosine similarity
4. **Augmentation** → Combine query + retrieved context
5. **Generation** → ChatGPT generates contextual response

## 🛠️ Setup & Installation

### 1. Prerequisites
```bash
# Required software
- Python 3.8+
- Zilliz Cloud account
- OpenAI API key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Zilliz Cloud Vector Database

1. **Sign up for Zilliz Cloud**:
   - Go to https://cloud.zilliz.com/
   - Create an account and new cluster
   - Note your cluster endpoint and API key

2. **Configure Zilliz Cloud connection**:
   ```env
   # Add to your .env file
   ZILLIZ_CLOUD_URI=https://your-cluster-endpoint.zillizcloud.com
   ZILLIZ_API_KEY=your_api_key_here
   ```

### 4. Environment Configuration
Create a `.env` file with your configuration:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Zilliz Cloud Configuration
ZILLIZ_CLOUD_URI=https://your-cluster-endpoint.zillizcloud.com
ZILLIZ_API_KEY=your_zilliz_api_key_here
```

**Getting your credentials:**
- **OpenAI API Key**: https://platform.openai.com/api-keys
- **Zilliz Cloud**: https://cloud.zilliz.com/ → Create cluster → Get endpoint & API key

### 5. Run the Application
```bash
# Start the FastAPI server
python main.py

# Or use uvicorn directly with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🚀 Usage Guide

### Getting Started
1. **Open your browser** to http://localhost:8000
2. **Load sample data** by clicking "📚 Load Sample Data"
3. **Start chatting** - Ask questions about the loaded documents stored in Zilliz Cloud
4. **Upload your own documents** using the upload buttons

### Chat Interface Features
- **Contextual Q&A** - Ask questions about documents stored in your Zilliz Cloud cluster
- **Source Attribution** - See which documents informed each response
- **File Upload** - Drag & drop `.txt` or `.md` files to store in Zilliz Cloud
- **Manual Entry** - Add documents through the web form

### Example Queries
```
"What is FastAPI and how does it work?"
"Tell me about vector databases and Zilliz Cloud"
"How do I implement similarity search with Zilliz Cloud?"
"What are the benefits of RAG systems using cloud infrastructure?"
```

## 📡 API Endpoints

### Chat & RAG
```http
POST /ask
Content-Type: application/json
{
  "message": "Your question here"
}

Response:
{
  "response": "AI generated answer with context",
  "sources": ["Document Title 1", "Document Title 2"]
}
```

### Document Management
```http
# Add document via JSON
POST /documents
{
  "title": "Document Title",
  "content": "Document content here..."
}

# Upload file
POST /documents/upload
Content-Type: multipart/form-data
file: (text file)

# Search documents
GET /documents/search?query=your_search_term&limit=5

# Initialize sample data
POST /init-sample-data
```

### System Status
```http
GET /health
Response:
{
  "status": "healthy",
  "connection_type": "Zilliz Cloud",
  "milvus_connected": true,
  "embedding_model": "text-embedding-3-small"
}
```

## 🔧 Technical Details

### Vector Database Schema
```python
Collection: knowledge_base (stored in Zilliz Cloud)
Fields:
- id: VARCHAR(100) [Primary Key]
- title: VARCHAR(500) 
- content: VARCHAR(5000)
- embedding: FLOAT_VECTOR(1536) [Searchable]

Index: IVF_FLAT with COSINE similarity
```

### Embedding Model
- **Model**: `text-embedding-3-small` (OpenAI)
- **Dimensions**: 1536
- **Similarity Metric**: Cosine Similarity
- **Performance**: High-quality, consistent with OpenAI ecosystem

### RAG Configuration
- **Retrieval**: Top-3 most similar documents
- **Context Window**: Up to 5000 chars per document
- **Generation Model**: GPT-3.5-turbo
- **Embedding Model**: text-embedding-3-small
- **Temperature**: 0.7 (balanced creativity/accuracy)
- **Vector Database**: Zilliz Cloud (managed, auto-scaling)

## 🐛 Troubleshooting

### Common Issues

#### Zilliz Cloud Connection Issues
```bash
# Verify your Zilliz Cloud credentials
echo $ZILLIZ_CLOUD_URI
echo $ZILLIZ_API_KEY

# Check if your cluster is active in Zilliz Cloud console
# Visit: https://cloud.zilliz.com/

# Test connection with health endpoint
curl http://localhost:8000/health
```

#### OpenAI API Issues
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API access
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### Zilliz Cloud Account Issues
```bash
# Common solutions:
- Ensure your Zilliz Cloud cluster is running (not paused)
- Verify your cluster endpoint URL is correct
- Check that your API key hasn't expired
- Confirm your account has sufficient credits/quota
```

## 📋 Dependencies

```
fastapi==0.116.1          # Web framework
uvicorn==0.35.0           # ASGI server
python-multipart==0.0.9   # File upload support
openai==1.40.0            # OpenAI API client (chat + embeddings)
pymilvus==2.5.13          # Zilliz Cloud vector database client
marshmallow==3.19.0       # Data serialization (pymilvus dependency)
environs==9.5.0           # Environment configuration (pymilvus dependency)
pandas==2.2.3             # Data processing
numpy==1.26.4             # Numerical computing
python-dotenv==1.0.1      # Environment variables
```

## 🎯 Future Enhancements

- [ ] **Multi-modal Support** - Images, PDFs, videos
- [ ] **Advanced Chunking** - Smart text segmentation
- [ ] **Hybrid Search** - Keyword + vector search
- [ ] **User Authentication** - Personal knowledge bases
- [ ] **Document Versioning** - Track document updates
- [ ] **Analytics Dashboard** - Usage metrics and insights
- [ ] **API Rate Limiting** - Production-ready controls
- [ ] **Bulk Import** - CSV, JSON data sources

## 📄 License

MIT License - Feel free to use this project for learning and development!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review Zilliz Cloud documentation: https://docs.zilliz.com/
- OpenAI API documentation: https://platform.openai.com/docs
- Zilliz Cloud console: https://cloud.zilliz.com/ 