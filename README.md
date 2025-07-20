# UWS WhatsApp Assistant Bot

Advanced AI-powered WhatsApp assistant for University of the West of Scotland (UWS) students, built with CrewAI and FastAPI.

## Features

- **Advanced RAG Pipeline**: Semantic search with query rewriting and optimization
- **Multi-LLM Support**: OpenAI and Groq models with easy switching
- **Memory Integration**: Conversation history storage and retrieval via MCP
- **HubSpot Integration**: Contact management and personalized responses via MCP
- **WhatsApp Optimized**: Responses under 1500 characters
- **University-Focused**: Only answers UWS-related queries
- **Semantic Search**: 384-dimension embeddings with Pinecone

## Supported Models

### OpenAI Models
- `gpt-4-turbo-preview`
- `gpt-4`
- `gpt-3.5-turbo`
- `gpt-4o`
- `gpt-4o-mini` (default)

### Groq Models
- `llama3-70b-8192`
- `llama3-8b-8192` (default)
- `mixtral-8x7b-32768`
- `gemma-7b-it`

## MCP Tools Integration

The bot uses Model Context Protocol (MCP) to access:

### Memory MCP Tools
- `mem0-mcp-stdio:search_memories` - Search conversation history
- `mem0-mcp-stdio:create_memory` - Store new conversations
- `mem0-mcp-stdio:get_all_memories` - Retrieve all user memories
- `mem0-mcp-stdio:update_memory` - Update existing memories

### HubSpot MCP Tools
- `hubspot_search_contacts` - Find contact information
- `hubspot_schedule_link` - Generate scheduling links
- `hubspot_application_status` - Check application status

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/sagar-jg/cloud-ma.git
cd cloud-ma
```

### 2. Environment Configuration
```bash
cp .env.template .env
# Edit .env with your API keys and MCP paths
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run with Docker
```bash
docker-compose up -d
```

### 5. Run Locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Chat
```http
POST /chat
{
  "user_id": "student123",
  "message": "What are the admission requirements for Computer Science?",
  "timestamp": "2024-01-01T10:00:00Z"
}
```

### Configure Model
```http
POST /configure
{
  "provider": "openai",
  "model": "gpt-4o-mini"
}
```

### List Models
```http
GET /models
```

### Health Check
```http
GET /health
```

## Architecture

### Two-Agent System
1. **UWS Query Analyst**: Validates UWS relevance and determines strategy
2. **UWS Information Assistant**: Handles retrieval, personalization, and response formatting

### RAG Pipeline Features
- **Query Analysis**: Intent detection and rewriting
- **Semantic Search**: 384-dimension embeddings with Pinecone
- **Relevance Filtering**: Score-based document filtering (>0.7)
- **Memory Integration**: Context-aware responses
- **MCP Tool Access**: Direct integration with existing MCP servers

## Configuration

### MCP Server Paths
Update these paths in the Config class:
```python
# HubSpot MCP
"/path/to/hubspot-mcp/hubspot_server.py"

# Memory MCP  
"/path/to/mem0-mcp/mcp_stdio.py"
```

### Pinecone Setup
- Index: `interna`
- Dimensions: 384
- Encoder: `all-MiniLM-L6-v2`

## Error Handling

- Fallback messages for unknown queries
- Context filtering for non-university topics
- Graceful MCP tool failures
- Comprehensive logging

## Intent Filtering

**Valid Queries**: courses, admissions, fees, scholarships, campus, applications
**Invalid Queries**: politics, weather, movies, general non-university topics

## Response Format

- Maximum 1500 characters for WhatsApp
- Human-like tone (not AI-like)
- Includes relevant links when available
- Stores every conversation in memory
- Personalizes using HubSpot data when needed

## Logging

All operations logged with:
- User IDs and timestamps
- Query processing steps
- Tool execution results
- Error handling details

## License

MIT License