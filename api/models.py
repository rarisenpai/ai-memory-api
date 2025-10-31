from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

# ==================== Base Response Model ====================

class BaseResponse(BaseModel):
    success: bool
    error: Optional[str] = None

# ==================== Block Definition ====================

class BlockDefinition(BaseModel):
    """Definition for a memory block"""
    label: str = Field(..., description="Block identifier (e.g., 'human', 'policies', 'preferences')")
    description: str = Field(..., description="What this block stores and when to update it")
    value: str = Field(default="", description="Initial content")
    char_limit: int = Field(default=10000, description="Maximum characters")

# ==================== Initialize User Models ====================

class InitializeUserRequest(BaseModel):
    """Initialize memory with default blocks for a customer"""
    org_id: str = Field(..., description="Organization identifier")
    customer_phone: str = Field(..., description="Customer phone number")
    user_info: str = Field(default="", description="Initial user information")
    additional_tags: Optional[List[str]] = Field(
        default=None, 
        description="Optional custom tags (e.g., ['vip', 'premium-tier'])"
    )
    reset: bool = Field(default=False, description="Delete and recreate if agent exists")

class InitializeUserResponse(BaseResponse):
    message: Optional[str] = None
    agent_id: Optional[str] = None
    tags: Optional[List[str]] = Field(default=None, description="Tags assigned to the agent")

# ==================== Initialize with Blocks Models ====================

class InitializeWithBlocksRequest(BaseModel):
    """Initialize memory with custom blocks"""
    org_id: str = Field(..., description="Organization identifier")
    customer_phone: str = Field(..., description="Customer phone number")
    blocks: List[BlockDefinition]
    additional_tags: Optional[List[str]] = Field(
        default=None, 
        description="Optional custom tags"
    )
    reset: bool = Field(default=False, description="Delete and recreate if agent exists")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "org_id": "tamambo-restaurant",
                    "customer_phone": "254743637047",
                    "blocks": [
                        {
                            "label": "customer_profile",
                            "description": "Basic customer info",
                            "value": "Premium subscriber",
                            "char_limit": 5000
                        },
                        {
                            "label": "policies",
                            "description": "Company policies",
                            "value": "Refund: 30 days",
                            "char_limit": 10000
                        }
                    ],
                    "additional_tags": ["vip", "premium-tier"],
                    "reset": False
                }
            ]
        }
    }

class InitializeWithBlocksResponse(BaseResponse):
    message: Optional[str] = None
    agent_id: Optional[str] = None
    blocks_created: Optional[int] = None
    tags: Optional[List[str]] = Field(default=None, description="Tags assigned to the agent")

# ==================== Add Conversation Models ====================

class AddConversationRequest(BaseModel):
    org_id: str = Field(..., description="Organization identifier")
    customer_phone: str = Field(..., description="Customer phone number")
    messages: List[Dict[str, str]] = Field(
        ..., 
        description="List of message objects with 'role' and 'content'"
    )
    store_in_archival: bool = Field(
        default=True, 
        description="Enable semantic search on these messages"
    )
    wait_for_completion: bool = Field(
        default=True, 
        description="Block until sleeptime agent finishes processing"
    )

class AddConversationResponse(BaseResponse):
    run_id: Optional[str] = None
    message: Optional[str] = None
    messages_count: Optional[int] = None

# ==================== Context Models ====================

class ContextResponse(BaseResponse):
    context: str = ""
    message: Optional[str] = None

class SummaryResponse(BaseResponse):
    summary: str = ""
    message: Optional[str] = None

class SearchResult(BaseModel):
    success: bool
    results: List[str] = []
    count: int = 0
    total_found: Optional[int] = None
    error: Optional[str] = None

class FullContextResponse(BaseResponse):
    user_context: str = ""
    summary: str = ""
    relevant_memories: List[str] = []
    combined_context: str = ""

# ==================== Delete and Agent Models ====================

class DeleteResponse(BaseResponse):
    message: Optional[str] = None

class AgentIdResponse(BaseResponse):
    agent_id: Optional[str] = None
    dashboard_url: Optional[str] = None
    message: Optional[str] = None

# ==================== NEW: Tag Management Models ====================

class AddTagsRequest(BaseModel):
    """Request to add tags to a customer's agent"""
    tags: List[str] = Field(..., description="Tags to add (e.g., ['vip', 'premium-tier'])")

class RemoveTagsRequest(BaseModel):
    """Request to remove tags from a customer's agent"""
    tags: List[str] = Field(..., description="Tags to remove")

class TagsResponse(BaseResponse):
    agent_id: Optional[str] = None
    tags: Optional[List[str]] = None
    added: Optional[List[str]] = Field(default=None, description="Tags that were added")
    removed: Optional[List[str]] = Field(default=None, description="Tags that were removed")
    protected: Optional[List[str]] = Field(
        default=None, 
        description="Protected tags that could not be removed"
    )

class GetTagsResponse(BaseResponse):
    agent_id: Optional[str] = None
    tags: Optional[List[str]] = None

# ==================== NEW: Organization Management Models ====================

class AgentInfo(BaseModel):
    """Information about a sleeptime agent"""
    agent_id: str
    name: str
    tags: List[str]
    created_at: Optional[str] = None

class ListOrgAgentsResponse(BaseResponse):
    org_id: str
    total: int
    agents: List[AgentInfo] = []

class OrgStatsResponse(BaseResponse):
    org_id: str
    total_agents: int
    tag_distribution: Dict[str, int] = Field(
        default={}, 
        description="Count of each custom tag across all agents"
    )

# ==================== Health Models ====================

class HealthResponse(BaseModel):
    status: str
    service: str

class DetailedHealthResponse(BaseModel):
    status: str
    service: str
    components: Dict[str, str] = Field(default={}, description="Status of each component")
    letta_connection: bool = Field(default=False, description="Letta server connectivity")
    model: Optional[str] = None
    embedding: Optional[str] = None
    base_url: Optional[str] = None

class LettaHealthResponse(BaseModel):
    connected: bool
    base_url: Optional[str] = None
    agent_count: Optional[int] = None
    model: Optional[str] = None
    embedding: Optional[str] = None
    error: Optional[str] = None