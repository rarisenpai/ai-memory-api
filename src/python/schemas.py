from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Message(BaseModel): 
    """ Represents a message in the conversation """
    id: int
    agent_id: str
    content: str
    processed: bool
    role: Optional[str] = None
    name: Optional[str] = None
    registered_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class File(BaseModel):
    """ Represents a file in the database """
    id: int
    agent_id: str
    file_path: str
    file_hash: str
    size: int
    last_modified: datetime
    processed: bool
    label: str
    description: str
    registered_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None


class MessageCreate(BaseModel):
    """ Data for creating a new message """
    content: str
    role: str
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

