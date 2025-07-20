import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from enum import Enum

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import tool

# LLM imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Vector store imports
import pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"

class OpenAIModels(str, Enum):
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

class GroqModels(str, Enum):
    LLAMA3_70B = "llama3-70b-8192"
    LLAMA3_8B = "llama3-8b-8192"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA_7B = "gemma-7b-it"

# Request/Response models
class WhatsAppMessage(BaseModel):
    user_id: str
    message: str
    timestamp: Optional[datetime] = None

class BotResponse(BaseModel):
    response: str
    status: str
    user_id: str

# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.hubspot_api_key = os.getenv("HUBSPOT_API_KEY")
        
        # Default model settings
        self.model_provider = ModelProvider.OPENAI
        self.openai_model = OpenAIModels.GPT_4O_MINI
        self.groq_model = GroqModels.LLAMA3_8B
        
        # Pinecone settings
        self.pinecone_index_name = "interna"
        self.pinecone_dimension = 384
        
        # Response settings
        self.max_response_length = 1500
        
        # MCP server configurations
        self.server_params_list = [
            StdioServerParameters(
                command="python3",
                args=["/Users/sagargurav/Desktop/Sagar/Work/pinecone-mcp/hubspot-mcp/hubspot_server.py"],
                env={"HUBSPOT_API_KEY": self.hubspot_api_key},
            ),
            StdioServerParameters(
                command="/Users/sagargurav/Desktop/Sagar/mem0-mcp/mem0-mcp/mem0-mcp/venv/bin/python3",
                args=["/Users/sagargurav/Desktop/Sagar/mem0-mcp/mem0-mcp/mem0-mcp/mcp_stdio.py"],
                env={
                    "OPENAI_API_KEY": self.openai_api_key,
                    "POSTGRES_HOST": "localhost",
                    "POSTGRES_PORT": "5432",
                    "POSTGRES_DB": "stage_genai",
                    "POSTGRES_USER": "postgres",
                    "POSTGRES_PASSWORD": "postgres",
                    "MCP_SESSION_SECRET": "your-secure-session-secret",
                    "MCP_RATE_LIMIT": "60",
                    "MCP_LOG_LEVEL": "INFO"
                },
            ),
        ]

config = Config()

# Advanced RAG Pipeline
class AdvancedRAGPipeline:
    def __init__(self):
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(config.pinecone_index_name)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
        logger.info("RAG Pipeline initialized with Pinecone and SentenceTransformer")
    
    async def analyze_and_rewrite_query(self, query: str, memory_context: str = "") -> str:
        """Analyze query and rewrite for better retrieval"""
        try:
            llm = self._get_llm()
            
            analysis_prompt = f"""
            You are a query analysis expert for University of the West of Scotland (UWS).
            
            Original Query: {query}
            Memory Context: {memory_context}
            
            Analyze this query and rewrite it for optimal semantic search. Consider:
            1. Extract key educational concepts and UWS-specific terms
            2. Expand abbreviations and add relevant synonyms
            3. Include context from memory if relevant
            4. Focus on university-related intent (courses, admissions, fees, scholarships, campus, etc.)
            
            Return ONLY the rewritten query, nothing else.
            """
            
            response = await llm.ainvoke(analysis_prompt)
            rewritten_query = response.content.strip()
            logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search on Pinecone"""
        try:
            # Encode query
            query_embedding = self.encoder.encode(query).tolist()
            
            # Search Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process results
            documents = []
            for match in search_results.matches:
                if match.score > 0.7:  # Relevance threshold
                    doc = {
                        'content': match.metadata.get('text', ''),
                        'source': match.metadata.get('source', ''),
                        'score': match.score,
                        'metadata': match.metadata
                    }
                    documents.append(doc)
            
            logger.info(f"Found {len(documents)} relevant documents for query: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _get_llm(self):
        """Get configured LLM instance"""
        if config.model_provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=config.openai_model.value,
                api_key=config.openai_api_key,
                temperature=0.1
            )
        else:
            return ChatGroq(
                model=config.groq_model.value,
                api_key=config.groq_api_key,
                temperature=0.1
            )

# MCP Tools
class MCPManager:
    def __init__(self):
        self.sessions = {}
        logger.info("MCP Manager initialized")
    
    async def initialize_sessions(self):
        """Initialize MCP sessions"""
        try:
            for i, server_params in enumerate(config.server_params_list):
                session_name = f"session_{i}"
                session = await stdio_client(server_params)
                self.sessions[session_name] = session
                logger.info(f"MCP session {session_name} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MCP sessions: {e}")
    
    async def get_memory_context(self, user_id: str) -> str:
        """Get memory context for user"""
        try:
            if "session_1" in self.sessions:  # Memory MCP
                session = self.sessions["session_1"]
                result = await session.call_tool("mem0-mcp-stdio:search_memories", {
                    "query": f"user {user_id} conversation history",
                    "user_id": user_id,
                    "limit": 5
                })
                
                memories = json.loads(result.content)
                context = " ".join([mem.get('text', '') for mem in memories])
                logger.info(f"Retrieved memory context for user {user_id}")
                return context
        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
        return ""
    
    async def store_memory(self, user_id: str, message: str, response: str):
        """Store conversation in memory"""
        try:
            if "session_1" in self.sessions:  # Memory MCP
                session = self.sessions["session_1"]
                messages = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response}
                ]
                
                await session.call_tool("mem0-mcp-stdio:create_memory", {
                    "messages": messages,
                    "user_id": user_id,
                    "metadata": {"timestamp": datetime.now().isoformat()}
                })
                logger.info(f"Stored memory for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
    
    async def get_hubspot_contact(self, query: str) -> str:
        """Get HubSpot contact information"""
        try:
            if "session_0" in self.sessions:  # HubSpot MCP
                session = self.sessions["session_0"]
                result = await session.call_tool("hubspot_search_contacts", {
                    "query": query
                })
                return result.content
        except Exception as e:
            logger.error(f"Failed to get HubSpot contact: {e}")
        return ""

# CrewAI Agents
class UWSAssistantCrew:
    def __init__(self):
        self.llm = self._get_llm()
        self.rag_pipeline = AdvancedRAGPipeline()
        self.mcp_manager = MCPManager()
        
    def _get_llm(self):
        """Get configured LLM"""
        if config.model_provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=config.openai_model.value,
                api_key=config.openai_api_key,
                temperature=0.3
            )
        else:
            return ChatGroq(
                model=config.groq_model.value,
                api_key=config.groq_api_key,
                temperature=0.3
            )
    
    def create_crew(self) -> Crew:
        # Query Analyzer Agent  
        query_analyst = Agent(
            role="UWS Query Analyst",
            goal="Analyze student queries for UWS relevance and determine response strategy",
            backstory="""You are an expert at understanding University of the West of Scotland student queries.
            You can determine if queries are university-related (courses, admissions, fees, scholarships, campus, applications)
            or off-topic (politics, weather, movies). You guide the response process.""",
            llm=self.llm,
            verbose=True
        )
        
        # Information Retrieval & Response Agent with direct MCP access
        information_agent = Agent(
            role="UWS Information Assistant", 
            goal="Retrieve relevant information and create personalized WhatsApp responses under 1500 characters",
            backstory="""You are a helpful UWS student assistant. You can search documents, access memory,
            and find contact information to provide comprehensive, personalized responses.
            You format responses for WhatsApp messaging with proper links and human-like tone.""",
            llm=self.llm,
            verbose=True
        )
        
        return Crew(
            agents=[query_analyst, information_agent],
            process=Process.sequential,
            verbose=True
        )
    
    def _search_documents(self, query: str) -> str:
        """Search UWS documents using advanced RAG pipeline"""
        try:
            loop = asyncio.get_event_loop()
            documents = loop.run_until_complete(self.rag_pipeline.semantic_search(query))
            
            if not documents:
                return "No relevant documents found for your query."
            
            # Format documents for response
            formatted_docs = []
            for doc in documents[:3]:  # Top 3 most relevant
                content = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                source = doc.get('source', 'UWS Document')
                formatted_docs.append(f"Source: {source}\nContent: {content}")
            
            return "\n\n".join(formatted_docs)
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return "Unable to search documents at the moment."
    
    async def process_message(self, user_id: str, message: str) -> str:
        """Process WhatsApp message and generate response"""
        try:
            # Get memory context first
            memory_context = await self.mcp_manager.get_memory_context(user_id)
            logger.info(f"Retrieved memory context for user {user_id}: {len(memory_context)} chars")
            
            # Rewrite query for better search
            optimized_query = await self.rag_pipeline.analyze_and_rewrite_query(message, memory_context)
            logger.info(f"Optimized query: {optimized_query}")
            
            # Create tasks for 2-agent workflow
            tasks = [
                Task(
                    description=f"""
                    Analyze this student query: "{message}"
                    
                    Determine:
                    1. Is this a valid UWS-related query? (courses, admissions, fees, scholarships, campus, application status, etc.)
                    2. If it's about politics, weather, movies, or other non-university topics, mark as INVALID
                    3. What type of information strategy is needed?
                    
                    Provide analysis: VALID/INVALID and recommended approach.
                    """,
                    agent=self.create_crew().agents[0],
                    expected_output="Query validity analysis and response strategy"
                ),
                
                Task(
                    description=f"""
                    Based on the query analysis, provide a comprehensive response for: "{message}"
                    User ID: {user_id}
                    
                    Steps to follow:
                    1. Use MCP tools directly via the MCP manager:
                       - Search memory: session_1 -> mem0-mcp-stdio:search_memories 
                       - Search documents: Use RAG pipeline for semantic search
                       - Get contacts: session_0 -> hubspot_search_contacts (if personalization needed)
                       - Store memory: session_1 -> mem0-mcp-stdio:create_memory
                    
                    2. Create response under {config.max_response_length} characters
                    3. Include relevant links if available  
                    4. Use human-like, not AI-like language
                    
                    If query is INVALID (non-UWS), respond: "I can only help with UWS-related questions like courses, admissions, fees, and campus information."
                    
                    If no relevant info found: "I don't have specific information about that. Please contact UWS Student Services for assistance."
                    """,
                    agent=self.create_crew().agents[1],
                    expected_output="Complete WhatsApp-ready response with memory storage"
                )
            ]
            
            # Execute crew
            crew = self.create_crew()
            crew.tasks = tasks
            result = crew.kickoff()
            
            # Ensure response length
            response = str(result)
            if len(response) > config.max_response_length:
                response = response[:config.max_response_length-3] + "..."
            
            logger.info(f"Generated response for user {user_id}: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact UWS Student Services for assistance."

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting UWS WhatsApp Assistant Bot")
    mcp_manager = MCPManager()
    await mcp_manager.initialize_sessions()
    app.state.mcp_manager = mcp_manager
    app.state.assistant_crew = UWSAssistantCrew()
    app.state.assistant_crew.mcp_manager = mcp_manager
    yield
    # Shutdown
    logger.info("Shutting down UWS WhatsApp Assistant Bot")

app = FastAPI(
    title="UWS WhatsApp Assistant Bot",
    description="Advanced AI assistant for University of the West of Scotland students",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/chat", response_model=BotResponse)
async def chat_endpoint(message: WhatsAppMessage, background_tasks: BackgroundTasks):
    """Main chat endpoint for WhatsApp messages"""
    try:
        logger.info(f"Received message from user {message.user_id}: {message.message}")
        
        # Process message
        response = await app.state.assistant_crew.process_message(
            message.user_id, 
            message.message
        )
        
        return BotResponse(
            response=response,
            status="success",
            user_id=message.user_id
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return BotResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again or contact UWS Student Services for assistance.",
            status="error",
            user_id=message.user_id
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/configure")
async def configure_model(provider: ModelProvider, model: str):
    """Configure LLM model"""
    try:
        config.model_provider = provider
        if provider == ModelProvider.OPENAI:
            config.openai_model = OpenAIModels(model)
        else:
            config.groq_model = GroqModels(model)
        
        logger.info(f"Model configured: {provider.value} - {model}")
        return {"status": "success", "provider": provider.value, "model": model}
    except Exception as e:
        logger.error(f"Model configuration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "openai_models": [model.value for model in OpenAIModels],
        "groq_models": [model.value for model in GroqModels],
        "current_provider": config.model_provider.value,
        "current_model": config.openai_model.value if config.model_provider == ModelProvider.OPENAI else config.groq_model.value
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)