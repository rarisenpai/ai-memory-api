from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# Base Response Model
class BaseResponse(BaseModel):
    success: bool
    error: Optional[str] = None


# Block Definition
class BlockDefinition(BaseModel):
    """Definition for a memory block"""
    label: str = Field(..., description="Block identifier (e.g., 'human', 'policies', 'preferences')")
    description: str = Field(..., description="What this block stores and when to update it")
    value: str = Field(default="", description="Initial content")
    char_limit: int = Field(default=10000, description="Maximum characters")


# Initialize User Request/Response Models
class InitializeUserRequest(BaseModel):
    """Initialize memory with default blocks"""
    user_id: str
    user_info: str = ""
    reset: bool = False


class InitializeUserResponse(BaseResponse):
    message: Optional[str] = None
    agent_id: Optional[str] = None


# Initialize with Blocks Request/Response Models
class InitializeWithBlocksRequest(BaseModel):
    """Initialize memory with custom blocks"""
    user_id: str
    blocks: List[BlockDefinition]
    reset: bool = False
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "customer_123",
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
                    "reset": False
                }
            ]
        }
    }


class InitializeWithBlocksResponse(BaseResponse):
    message: Optional[str] = None
    agent_id: Optional[str] = None
    blocks_created: Optional[int] = None


# Add Conversation Request/Response Models
class AddConversationRequest(BaseModel):
    user_id: str
    messages: List[Dict[str, str]] = Field(..., description="List of message objects with 'role' and 'content'")
    store_in_archival: bool = True
    wait_for_completion: bool = True


class AddConversationResponse(BaseResponse):
    run_id: Optional[str] = None
    message: Optional[str] = None
    messages_count: Optional[int] = None


# Context Request/Response Models
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


# Delete and Agent Response Models
class DeleteResponse(BaseResponse):
    message: Optional[str] = None


class AgentIdResponse(BaseResponse):
    agent_id: Optional[str] = None
    dashboard_url: Optional[str] = None
    message: Optional[str] = None


# Health Response Model
class HealthResponse(BaseModel):
    status: str
    service: str