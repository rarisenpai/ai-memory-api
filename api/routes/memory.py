from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional
from ..models import (
    InitializeUserRequest,
    InitializeUserResponse,
    InitializeWithBlocksRequest,
    InitializeWithBlocksResponse,
    AddConversationRequest,
    AddConversationResponse,
    ContextResponse,
    SummaryResponse,
    SearchResult,
    FullContextResponse,
    DeleteResponse,
    AgentIdResponse,
    HealthResponse,
)
from ..memory_service import MemoryService
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])

# Initialize memory service with model configuration
memory_service = MemoryService(
    api_key=os.getenv("LETTA_API_KEY"),
    base_url=os.getenv("LETTA_BASE_URL"),
    model=os.getenv("LETTA_MODEL", "openai/gpt-4.1"),
    embedding=os.getenv("LETTA_EMBEDDING", "openai/text-embedding-3-small")
)


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Basic health check - returns service status
    
    Use this for:
    - Container orchestration (Docker, Kubernetes)
    - Load balancers
    - Monitoring systems
    - Uptime checks
    
    Returns 200 OK if service is running
    """
    return {
        "status": "healthy",
        "service": "Letta Memory API"
    }


@router.get("/health/detailed", tags=["health"])
async def detailed_health_check(
    check_letta: bool = Query(True, description="Test Letta connection")
):
    """
    Detailed health check with optional Letta connectivity test
    
    Returns:
    - Service status (healthy/degraded)
    - Component checks (API, Letta)
    - Configuration info (model, embedding, base_url)
    
    Status codes:
    - 200: All systems operational
    - 503: Letta connection failed (degraded state)
    """
    health_info = {
        "status": "healthy",
        "service": "Letta Memory API",
        "checks": {
            "api": "ok"
        },
        "config": {
            "model": memory_service.model,
            "embedding": memory_service.embedding,
            "base_url": memory_service.base_url or "https://api.letta.com",
            "letta_connected": False
        }
    }
    
    # Test Letta connection if requested
    if check_letta:
        try:
            # Lightweight operation - just list one agent
            agents = memory_service.memory.letta_client.agents.list(limit=1)
            health_info["checks"]["letta"] = "ok"
            health_info["config"]["letta_connected"] = True
            logger.info("Letta health check: OK")
        except Exception as e:
            health_info["status"] = "degraded"
            health_info["checks"]["letta"] = "failed"
            health_info["checks"]["letta_error"] = str(e)
            logger.error(f"Letta health check failed: {e}")
    
    # Set appropriate HTTP status code
    status_code = (
        status.HTTP_200_OK 
        if health_info["status"] == "healthy" 
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    
    return JSONResponse(
        status_code=status_code,
        content=health_info
    )


@router.get("/health/letta", tags=["health"])
async def letta_connection_check():
    """
    Specific Letta connectivity and configuration check
    
    Tests connection to Letta server and returns:
    - Connection status
    - Base URL
    - Agent count (if connected)
    - Model configuration
    
    Useful for debugging Letta connection issues
    """
    try:
        # Test connection
        agents = memory_service.memory.letta_client.agents.list(limit=1)
        agent_count = len(agents) if agents else 0
        
        return {
            "status": "connected",
            "service": "Letta",
            "base_url": memory_service.base_url or "https://api.letta.com",
            "agent_count": agent_count,
            "model": memory_service.model,
            "embedding": memory_service.embedding,
            "sdk_tag": memory_service.memory._default_tag
        }
    except Exception as e:
        logger.error(f"Letta connection check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "disconnected",
                "service": "Letta",
                "base_url": memory_service.base_url or "https://api.letta.com",
                "error": str(e)
            }
        )


# ============================================================================
# MEMORY INITIALIZATION ENDPOINTS
# ============================================================================

@router.post("/initialize", response_model=InitializeUserResponse)
async def initialize_user(request: InitializeUserRequest):
    """
    Initialize memory for a user with default blocks (human, summary)
    
    For custom blocks, use /initialize-with-blocks
    
    Example:
    ```json
    {
      "user_id": "alice",
      "user_info": "Software developer, works remotely",
      "reset": false
    }
    ```
    """
    result = memory_service.initialize_user(
        user_id=request.user_id,
        user_info=request.user_info,
        reset=request.reset
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


@router.post("/initialize-with-blocks", response_model=InitializeWithBlocksResponse)
async def initialize_with_blocks(request: InitializeWithBlocksRequest):
    """
    Initialize memory with custom blocks
    
    Use this for advanced use cases like:
    - Customer support bots (customer_profile, support_history, policies)
    - Personal assistants (preferences, goals, schedule)
    - Domain experts (knowledge_base, recent_updates, guidelines)
    
    Note: 'human' and 'summary' blocks are automatically added if not present
    
    Example:
    ```json
    {
      "user_id": "customer_123",
      "blocks": [
        {
          "label": "customer_profile",
          "description": "Customer tier, preferences, contact info",
          "value": "Premium tier, prefers email",
          "char_limit": 5000
        },
        {
          "label": "policies",
          "description": "Relevant company policies",
          "value": "30-day refund, 24/7 premium support",
          "char_limit": 10000
        }
      ],
      "reset": false
    }
    ```
    """
    # Convert Pydantic models to dicts
    blocks = [block.model_dump() for block in request.blocks]
    
    result = memory_service.initialize_with_blocks(
        user_id=request.user_id,
        blocks=blocks,
        reset=request.reset
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


# ============================================================================
# MEMORY UPDATE ENDPOINTS
# ============================================================================

@router.post("/add", response_model=AddConversationResponse)
async def add_conversation(request: AddConversationRequest):
    """
    Add conversation messages to memory
    
    The Letta sleeptime agent processes messages and updates memory blocks
    
    Parameters:
    - user_id: User identifier
    - messages: List of {"role": "user/assistant", "content": "..."}
    - store_in_archival: Enable semantic search (default: true)
    - wait_for_completion: Block until processing done (default: true)
    
    Example:
    ```json
    {
      "user_id": "alice",
      "messages": [
        {"role": "user", "content": "I finished the Python project!"},
        {"role": "assistant", "content": "Congratulations! How did it go?"}
      ],
      "store_in_archival": true,
      "wait_for_completion": true
    }
    ```
    """
    result = memory_service.add_conversation(
        messages=request.messages,
        user_id=request.user_id,
        store_in_archival=request.store_in_archival,
        wait_for_completion=request.wait_for_completion
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


# ============================================================================
# MEMORY RETRIEVAL ENDPOINTS
# ============================================================================

@router.get("/context", response_model=FullContextResponse)
async def get_full_context(
    user_id: str = Query(..., description="User identifier"),
    query: Optional[str] = Query(None, description="Optional query to search for relevant memories"),
    max_results: int = Query(3, description="Number of search results to include"),
    include_summary: bool = Query(True, description="Include conversation summary")
):
    """
    Get comprehensive context for injecting into LLM system prompts
    
    Returns:
    - user_context: The 'human' memory block
    - summary: Conversation summary block
    - relevant_memories: Semantic search results (if query provided)
    - combined_context: All above formatted for prompt injection
    
    Example usage in n8n:
    1. Call this endpoint: GET /memory/context?user_id=alice&query=current_project
    2. Inject combined_context into your LLM's system prompt
    3. Get context-aware responses!
    
    Example:
    ```
    GET /memory/context?user_id=alice&query=What%20is%20Alice%20working%20on&max_results=3
    ```
    """
    result = memory_service.get_full_context(
        current_query=query,
        user_id=user_id,
        max_search_results=max_results,
        include_summary=include_summary
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Context not found"))
    
    return result


@router.get("/user-context", response_model=ContextResponse)
async def get_user_context(
    user_id: str = Query(..., description="User identifier"),
    format: str = Query("xml", description="Format: 'xml' or 'raw'")
):
    """
    Get user memory block only
    
    Formats:
    - xml: <human description="...">content</human>
    - raw: content only
    
    Example:
    ```
    GET /memory/user-context?user_id=alice&format=xml
    ```
    """
    result = memory_service.get_user_context(user_id=user_id, format=format)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "User context not found"))
    
    return result


@router.get("/summary", response_model=SummaryResponse)
async def get_summary(
    user_id: str = Query(..., description="User identifier"),
    format: str = Query("xml", description="Format: 'xml' or 'raw'")
):
    """
    Get conversation summary block only
    
    Example:
    ```
    GET /memory/summary?user_id=alice&format=xml
    ```
    """
    result = memory_service.get_summary(user_id=user_id, format=format)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Summary not found"))
    
    return result


@router.get("/search", response_model=SearchResult)
async def search_memories(
    user_id: str = Query(..., description="User identifier"),
    query: str = Query(..., description="Search query"),
    max_results: int = Query(5, description="Maximum number of results"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by (e.g., 'user,assistant')")
):
    """
    Semantic search over conversation history
    
    Requires archival storage enabled when adding messages
    
    Tag filtering:
    - No tags: searches user messages only (default)
    - "user": user messages
    - "assistant": assistant messages
    - "": all messages (empty string)
    
    Example:
    ```
    GET /memory/search?user_id=alice&query=Python%20projects&max_results=5&tags=user
    ```
    """
    tags_list = None
    if tags:
        tags_list = [t.strip() for t in tags.split(',')]
    
    result = memory_service.search_memories(
        query=query,
        user_id=user_id,
        max_results=max_results,
        tags=tags_list
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
    
    return result


# ============================================================================
# MANAGEMENT ENDPOINTS
# ============================================================================

@router.delete("/user/{user_id}", response_model=DeleteResponse)
async def delete_user(user_id: str):
    """
    Delete all memory for a user
    
    This deletes:
    - The Letta agent
    - All memory blocks
    - All archival passages
    
    ⚠️ This operation is irreversible!
    
    Example:
    ```
    DELETE /memory/user/alice
    ```
    """
    result = memory_service.delete_user(user_id=user_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Delete failed"))
    
    return result


@router.get("/agent/{user_id}", response_model=AgentIdResponse)
async def get_agent_id(user_id: str):
    """
    Get Letta agent ID and dashboard URL for a user
    
    Useful for debugging - you can view the agent in Letta's ADE
    
    Returns:
    - agent_id: The Letta agent ID
    - dashboard_url: Direct link to view in Letta ADE
    
    Example:
    ```
    GET /memory/agent/alice
    ```
    
    Response:
    ```json
    {
      "success": true,
      "agent_id": "agent-abc123...",
      "dashboard_url": "https://app.letta.com/agents/agent-abc123..."
    }
    ```
    """
    result = memory_service.get_agent_id(user_id=user_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Agent not found"))
    
    return result