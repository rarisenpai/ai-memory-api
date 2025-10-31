from fastapi import APIRouter, Query, HTTPException, status, Path
from fastapi.responses import JSONResponse
from typing import Optional, List
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
    DetailedHealthResponse,
    LettaHealthResponse,
    # New models
    AddTagsRequest,
    RemoveTagsRequest,
    TagsResponse,
    GetTagsResponse,
    ListOrgAgentsResponse,
    OrgStatsResponse,
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


@router.get("/health/detailed", response_model=DetailedHealthResponse, tags=["health"])
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
        "components": {
            "api": "ok"
        },
        "letta_connection": False,
        "model": memory_service.model,
        "embedding": memory_service.embedding,
        "base_url": memory_service.base_url or "https://api.letta.com"
    }
    
    # Test Letta connection if requested
    if check_letta:
        try:
            # Lightweight operation - just list one agent
            agents = memory_service.memory.letta_client.agents.list(limit=1)
            health_info["components"]["letta"] = "ok"
            health_info["letta_connection"] = True
            logger.info("Letta health check: OK")
        except Exception as e:
            health_info["status"] = "degraded"
            health_info["components"]["letta"] = "failed"
            health_info["components"]["letta_error"] = str(e)
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


@router.get("/health/letta", response_model=LettaHealthResponse, tags=["health"])
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
            "connected": True,
            "base_url": memory_service.base_url or "https://api.letta.com",
            "agent_count": agent_count,
            "model": memory_service.model,
            "embedding": memory_service.embedding
        }
    except Exception as e:
        logger.error(f"Letta connection check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "connected": False,
                "base_url": memory_service.base_url or "https://api.letta.com",
                "error": str(e)
            }
        )


# ============================================================================
# MEMORY INITIALIZATION ENDPOINTS (Multi-Tenant)
# ============================================================================

@router.post(
    "/organizations/{org_id}/customers/{customer_phone}/initialize",
    response_model=InitializeUserResponse
)
async def initialize_customer(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    user_info: str = Query("", description="Initial user information"),
    additional_tags: Optional[List[str]] = Query(None, description="Custom tags (e.g., vip, premium)"),
    reset: bool = Query(False, description="Delete and recreate if exists")
):
    """
    Initialize memory for a customer within an organization
    
    Automatically creates tags:
    - ai-memory-sdk
    - org:{org_id}
    - customer:{customer_phone}
    - type:memory-agent
    - Plus any additional_tags
    
    Example:
```
    POST /memory/organizations/tamambo-restaurant/customers/254743637047/initialize?user_info=Regular%20customer&additional_tags=vip&additional_tags=premium-tier
```
    """
    result = memory_service.initialize_user(
        org_id=org_id,
        customer_phone=customer_phone,
        user_info=user_info,
        additional_tags=additional_tags or [],
        reset=reset
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


@router.post(
    "/organizations/{org_id}/customers/{customer_phone}/initialize-with-blocks",
    response_model=InitializeWithBlocksResponse
)
async def initialize_customer_with_blocks(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    request: InitializeWithBlocksRequest = None
):
    """
    Initialize memory with custom blocks for a customer
    
    Use this for specialized agents like:
    - Restaurant assistants (food_preferences, dietary_restrictions, order_patterns)
    - Customer support (customer_profile, support_history, policies)
    - Personal assistants (preferences, goals, schedule)
    
    Note: 'human' and 'summary' blocks are automatically added if not present
    
    Example:
```json
    {
      "blocks": [
        {
          "label": "customer_profile",
          "description": "Customer tier, preferences, contact info",
          "value": "Premium tier, prefers email",
          "char_limit": 5000
        },
        {
          "label": "food_preferences",
          "description": "Dietary restrictions and favorite dishes",
          "value": "Vegetarian, loves Thai food",
          "char_limit": 3000
        }
      ],
      "additional_tags": ["vip", "premium-tier"],
      "reset": false
    }
```
    """
    # Override org_id and customer_phone from path parameters
    blocks = [block.model_dump() for block in request.blocks]
    
    result = memory_service.initialize_with_blocks(
        org_id=org_id,
        customer_phone=customer_phone,
        blocks=blocks,
        additional_tags=request.additional_tags or [],
        reset=request.reset
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


# ============================================================================
# MEMORY UPDATE ENDPOINTS (Multi-Tenant)
# ============================================================================

@router.post(
    "/organizations/{org_id}/customers/{customer_phone}/messages",
    response_model=AddConversationResponse
)
async def add_customer_messages(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    request: AddConversationRequest = None
):
    """
    Add conversation messages to customer's memory
    
    The Letta sleeptime agent processes messages and updates memory blocks
    
    Parameters:
    - messages: List of {"role": "user/assistant", "content": "..."}
    - store_in_archival: Enable semantic search (default: true)
    - wait_for_completion: Block until processing done (default: true)
    
    Example:
```json
    {
      "messages": [
        {"role": "user", "content": "I'd like to order pizza"},
        {"role": "assistant", "content": "Great! What toppings would you like?"}
      ],
      "store_in_archival": true,
      "wait_for_completion": true
    }
```
    """
    result = memory_service.add_conversation(
        org_id=org_id,
        customer_phone=customer_phone,
        messages=request.messages,
        store_in_archival=request.store_in_archival,
        wait_for_completion=request.wait_for_completion
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result


# ============================================================================
# MEMORY RETRIEVAL ENDPOINTS (Multi-Tenant)
# ============================================================================

@router.get(
    "/organizations/{org_id}/customers/{customer_phone}/context",
    response_model=FullContextResponse
)
async def get_customer_context(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
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
    1. Call this endpoint: GET /memory/organizations/tamambo/customers/254743637047/context?query=recent%20orders
    2. Inject combined_context into your LLM's system prompt
    3. Get context-aware responses!
    """
    result = memory_service.get_full_context(
        org_id=org_id,
        customer_phone=customer_phone,
        current_query=query,
        max_search_results=max_results,
        include_summary=include_summary
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Context not found"))
    
    return result


@router.get(
    "/organizations/{org_id}/customers/{customer_phone}/user-context",
    response_model=ContextResponse
)
async def get_customer_user_context(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    format: str = Query("xml", description="Format: 'xml' or 'raw'")
):
    """
    Get user memory block only
    
    Formats:
    - xml: <human description="...">content</human>
    - raw: content only
    """
    result = memory_service.get_user_context(
        org_id=org_id,
        customer_phone=customer_phone,
        format=format
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "User context not found"))
    
    return result


@router.get(
    "/organizations/{org_id}/customers/{customer_phone}/summary",
    response_model=SummaryResponse
)
async def get_customer_summary(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    format: str = Query("xml", description="Format: 'xml' or 'raw'")
):
    """
    Get conversation summary block only
    """
    result = memory_service.get_summary(
        org_id=org_id,
        customer_phone=customer_phone,
        format=format
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Summary not found"))
    
    return result


@router.get(
    "/organizations/{org_id}/customers/{customer_phone}/search",
    response_model=SearchResult
)
async def search_customer_memories(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    query: str = Query(..., description="Search query"),
    max_results: int = Query(5, description="Maximum number of results"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by")
):
    """
    Semantic search over customer's conversation history
    
    Requires archival storage enabled when adding messages
    
    Tag filtering:
    - No tags: searches user messages only (default)
    - "user": user messages
    - "assistant": assistant messages
    - "": all messages (empty string)
    """
    tags_list = None
    if tags:
        tags_list = [t.strip() for t in tags.split(',')]
    
    result = memory_service.search_memories(
        org_id=org_id,
        customer_phone=customer_phone,
        query=query,
        max_results=max_results,
        tags=tags_list
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
    
    return result


# ============================================================================
# TAG MANAGEMENT ENDPOINTS (NEW)
# ============================================================================

@router.get(
    "/organizations/{org_id}/customers/{customer_phone}/tags",
    response_model=GetTagsResponse
)
async def get_customer_tags(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number")
):
    """
    Get all tags for a customer's sleeptime agent
    
    Example:
```
    GET /memory/organizations/tamambo/customers/254743637047/tags
```
    
    Response:
```json
    {
      "success": true,
      "agent_id": "agent-abc123...",
      "tags": [
        "ai-memory-sdk",
        "org:tamambo",
        "customer:254743637047",
        "type:memory-agent",
        "vip",
        "premium-tier"
      ]
    }
```
    """
    result = memory_service.get_agent_tags(
        org_id=org_id,
        customer_phone=customer_phone
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Agent not found"))
    
    return result


@router.post(
    "/organizations/{org_id}/customers/{customer_phone}/tags",
    response_model=TagsResponse
)
async def add_customer_tags(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    request: AddTagsRequest = None
):
    """
    Add tags to a customer's sleeptime agent
    
    Example:
```json
    {
      "tags": ["vip", "premium-tier", "frequent-orderer"]
    }
```
    """
    result = memory_service.add_tags(
        org_id=org_id,
        customer_phone=customer_phone,
        new_tags=request.tags
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to add tags"))
    
    return result


@router.delete(
    "/organizations/{org_id}/customers/{customer_phone}/tags",
    response_model=TagsResponse
)
async def remove_customer_tags(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number"),
    tags: List[str] = Query(..., description="Tags to remove")
):
    """
    Remove tags from a customer's sleeptime agent
    
    Protected tags cannot be removed:
    - ai-memory-sdk
    - org:{org_id}
    - customer:{customer_phone}
    - type:memory-agent
    
    Example:
```
    DELETE /memory/organizations/tamambo/customers/254743637047/tags?tags=trial&tags=temp
```
    """
    result = memory_service.remove_tags(
        org_id=org_id,
        customer_phone=customer_phone,
        tags_to_remove=tags
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to remove tags"))
    
    return result


# ============================================================================
# ORGANIZATION MANAGEMENT ENDPOINTS (NEW)
# ============================================================================

@router.get(
    "/organizations/{org_id}/agents",
    response_model=ListOrgAgentsResponse
)
async def list_org_agents(
    org_id: str = Path(..., description="Organization identifier"),
    limit: int = Query(50, le=100, description="Maximum number of agents to return"),
    additional_tags: Optional[List[str]] = Query(None, description="Additional tag filters")
):
    """
    List all sleeptime agents for an organization
    
    Optional tag filtering:
```
    GET /memory/organizations/tamambo/agents?additional_tags=vip&additional_tags=premium-tier
```
    
    Returns:
    - List of agents with their tags
    - Total count
    """
    result = memory_service.list_org_agents(
        org_id=org_id,
        limit=limit,
        additional_tags=additional_tags
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to list agents"))
    
    return result


@router.get(
    "/organizations/{org_id}/stats",
    response_model=OrgStatsResponse
)
async def get_org_stats(
    org_id: str = Path(..., description="Organization identifier")
):
    """
    Get statistics for an organization
    
    Returns:
    - Total agents (customers)
    - Tag distribution (how many agents have each custom tag)
    
    Example:
```
    GET /memory/organizations/tamambo/stats
```
    
    Response:
```json
    {
      "success": true,
      "org_id": "tamambo",
      "total_agents": 150,
      "tag_distribution": {
        "vip": 25,
        "premium-tier": 40,
        "frequent-orderer": 60
      }
    }
```
    """
    result = memory_service.get_org_stats(org_id=org_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to get stats"))
    
    return result


# ============================================================================
# MANAGEMENT ENDPOINTS (Multi-Tenant)
# ============================================================================

@router.delete(
    "/organizations/{org_id}/customers/{customer_phone}",
    response_model=DeleteResponse
)
async def delete_customer(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number")
):
    """
    Delete all memory for a customer
    
    This deletes:
    - The Letta sleeptime agent
    - All memory blocks
    - All archival passages
    
    ⚠️ This operation is irreversible!
    """
    result = memory_service.delete_user(
        org_id=org_id,
        customer_phone=customer_phone
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Delete failed"))
    
    return result


@router.get(
    "/organizations/{org_id}/customers/{customer_phone}/agent",
    response_model=AgentIdResponse
)
async def get_customer_agent_id(
    org_id: str = Path(..., description="Organization identifier"),
    customer_phone: str = Path(..., description="Customer phone number")
):
    """
    Get Letta sleeptime agent ID and dashboard URL for a customer
    
    Useful for debugging - you can view the agent in Letta's ADE
    
    Example:
```
    GET /memory/organizations/tamambo/customers/254743637047/agent
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
    result = memory_service.get_agent_id(
        org_id=org_id,
        customer_phone=customer_phone
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Agent not found"))
    
    return result


# ============================================================================
# BACKWARD COMPATIBILITY ENDPOINTS (DEPRECATED - for migration only)
# ============================================================================

@router.post("/initialize", response_model=InitializeUserResponse, deprecated=True)
async def initialize_user_legacy(request: InitializeUserRequest):
    """
    [DEPRECATED] Use /organizations/{org_id}/customers/{customer_phone}/initialize instead
    
    Legacy endpoint for backward compatibility during migration
    Expects user_id in format: "org_id:customer_phone"
    """
    # Parse legacy user_id format
    if ":" not in request.user_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id format. Expected 'org_id:customer_phone' or use new multi-tenant endpoints"
        )
    
    org_id, customer_phone = request.user_id.split(":", 1)
    
    result = memory_service.initialize_user(
        org_id=org_id,
        customer_phone=customer_phone,
        user_info=request.user_info,
        additional_tags=request.additional_tags or [],
        reset=request.reset
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result