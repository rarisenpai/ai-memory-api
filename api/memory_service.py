from ai_memory_sdk import Memory
from typing import List, Dict, Optional, Any
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryService:
    """
    High-level memory service for n8n integration with multi-tenant support
    
    Uses tags for organization + customer identification:
    - ai-memory-sdk (SDK identifier)
    - org:{org_id}
    - customer:{customer_phone}
    - type:memory-agent
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        subject_id: Optional[str] = None,
        model: str = "openai/gpt-4.1-mini",
        embedding: str = "openai/text-embedding-3-small"
    ):
        """
        Initialize Memory Service
        
        Args:
            api_key: Letta API key (or from LETTA_API_KEY env var)
            base_url: Letta server URL (or from LETTA_BASE_URL env var)
            subject_id: Optional subject binding for single-user mode
            model: LLM model to use (default: openai/gpt-4.1-mini)
            embedding: Embedding model to use (default: openai/text-embedding-3-small)
        """
        self.subject_id = subject_id
        self.model = model
        self.embedding = embedding
        
        # Store for later use when creating agents
        self.api_key = api_key or os.getenv("LETTA_API_KEY")
        self.base_url = base_url or os.getenv("LETTA_BASE_URL")
        
        self.memory = Memory(
            api_key=self.api_key,
            base_url=self.base_url,
            subject_id=subject_id
        )
        
        mode = "subject-scoped" if subject_id else "multi-user"
        logger.info(f"Memory service initialized in {mode} mode with model={model}, embedding={embedding}")
    
    def _get_user_id(self, user_id: Optional[str] = None) -> str:
        """Get effective user_id"""
        if user_id:
            return user_id
        if self.subject_id:
            return self.subject_id
        raise ValueError("user_id is required in multi-user mode")
    
    def _build_tags(
        self, 
        org_id: str, 
        customer_phone: str,
        additional_tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build tag list for multi-tenant agents
        
        Returns tags in format:
        - ai-memory-sdk
        - org:{org_id}
        - customer:{customer_phone}
        - type:memory-agent
        - {additional_tags}
        """
        user_id = f"{org_id}:{customer_phone}"
        
        # Start with Memory SDK's default tags
        base_tags = self.memory._subject_tags(user_id)
        
        # Add structured org and customer tags
        multi_tenant_tags = [
            f"org:{org_id}",
            f"customer:{customer_phone}",
            "type:memory-agent"
        ]
        
        # Combine all tags
        all_tags = base_tags + multi_tenant_tags
        
        # Add any additional custom tags
        if additional_tags:
            all_tags.extend(additional_tags)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in all_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags
    
    def initialize_user(
        self, 
        org_id: str,
        customer_phone: str,
        user_info: str = "",
        additional_tags: Optional[List[str]] = None,
        reset: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize memory for a customer within an organization
        
        Args:
            org_id: Organization identifier
            customer_phone: Customer phone number
            user_info: Initial user information
            additional_tags: Optional custom tags (e.g., ["vip", "premium-tier"])
            reset: If True, delete and recreate agent
        """
        try:
            user_id = f"{org_id}:{customer_phone}"
            tags = self._build_tags(org_id, customer_phone, additional_tags)
            
            # Check if agent already exists using org + customer tags
            agent = self.memory.letta_client.agents.list(
                tags=[f"org:{org_id}", f"customer:{customer_phone}"],
                tags_match_all=True,
                limit=1
            )
            
            if agent:
                agent = agent[0]
                if reset:
                    self.memory._delete_agent(agent.id)
                    agent = None
                else:
                    return {
                        "success": True,
                        "message": f"Memory already exists for {user_id}",
                        "agent_id": agent.id,
                        "tags": agent.tags
                    }
            
            # Create sleeptime agent with tags
            agent_id = self.memory.letta_client.agents.create(
                name=f"Memory Agent - {org_id}:{customer_phone}",
                model=self.model,
                embedding=self.embedding,
                agent_type="sleeptime_agent",
                tags=tags,
                memory_blocks=[
                    {
                        "label": "human",
                        "description": "Details about the human user you are speaking to.",
                        "value": user_info
                    },
                    {
                        "label": "summary",
                        "description": "A short (1-2 sentences) running summary of the conversation.",
                        "value": ""
                    }
                ]
            ).id
            
            # Create initial passage
            self.memory.letta_client.agents.passages.create(
                agent_id=agent_id,
                text=f"Initialized memory for user {user_id}",
                tags=[self.memory._default_tag]
            )
            
            logger.info(f"Initialized memory for {user_id} with tags: {tags}")
            return {
                "success": True, 
                "message": f"Initialized memory for {user_id}",
                "agent_id": agent_id,
                "tags": tags
            }
            
        except Exception as e:
            logger.error(f"Error initializing user: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def initialize_with_blocks(
        self,
        org_id: str,
        customer_phone: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        additional_tags: Optional[List[str]] = None,
        reset: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize memory with custom blocks for multi-tenant setup
        
        Args:
            org_id: Organization identifier
            customer_phone: Customer phone number
            blocks: Custom memory blocks
            additional_tags: Optional custom tags
            reset: If True, delete and recreate agent
        """
        try:
            user_id = f"{org_id}:{customer_phone}"
            tags = self._build_tags(org_id, customer_phone, additional_tags)
            
            # Check if agent exists
            agent = self.memory.letta_client.agents.list(
                tags=[f"org:{org_id}", f"customer:{customer_phone}"],
                tags_match_all=True,
                limit=1
            )
            
            if agent:
                agent = agent[0]
                if reset:
                    self.memory._delete_agent(agent.id)
                    agent = None
                else:
                    return {
                        "success": False,
                        "error": f"Memory already exists for {user_id}. Use reset=True to recreate.",
                        "agent_id": agent.id
                    }
            
            # Ensure blocks is a list
            if blocks is None:
                blocks = []
            
            # Get labels of blocks already provided by the user
            existing_labels = {block.get("label") for block in blocks if block.get("label")}
            
            # Define the default blocks that should always exist
            default_block_definitions = [
                {
                    "label": "human",
                    "description": "Information about the human user.",
                    "value": "",
                    "char_limit": 10000
                },
                {
                    "label": "summary",
                    "description": "A rolling summary of the conversation.",
                    "value": "",
                    "char_limit": 10000
                }
            ]
            
            # Add any missing default blocks
            for default_block in default_block_definitions:
                if default_block["label"] not in existing_labels:
                    blocks.append(default_block)
            
            # Format blocks for agent creation
            memory_blocks = []
            for block in blocks:
                memory_blocks.append({
                    "label": block.get("label"),
                    "description": block.get("description"),
                    "value": block.get("value", ""),
                    "limit": block.get("char_limit", 10000)
                })
            
            # Create sleeptime agent with tags
            agent_id = self.memory.letta_client.agents.create(
                name=f"Memory Agent - {user_id}",
                model=self.model,
                embedding=self.embedding,
                agent_type="sleeptime_agent",
                tags=tags,
                memory_blocks=memory_blocks
            ).id
            
            # Create initial passage
            self.memory.letta_client.agents.passages.create(
                agent_id=agent_id,
                text=f"Initialized memory for user {user_id}",
                tags=[self.memory._default_tag]
            )
            
            logger.info(f"Created agent with {len(memory_blocks)} blocks and tags: {tags}")
            return {
                "success": True,
                "message": f"Initialized memory for {user_id}",
                "agent_id": agent_id,
                "blocks_created": len(memory_blocks),
                "tags": tags
            }
            
        except Exception as e:
            logger.error(f"Error initializing with blocks: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_conversation(
        self, 
        org_id: str,
        customer_phone: str,
        messages: List[Dict[str, str]],
        store_in_archival: bool = True,
        wait_for_completion: bool = True
    ) -> Dict[str, Any]:
        """Add conversation messages to memory"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            
            # Ensure user exists
            init_result = self.initialize_user(org_id, customer_phone)
            if not init_result["success"] and "already exists" not in init_result.get("message", ""):
                return init_result
            
            logger.info(f"Adding {len(messages)} messages for {user_id}")
            run_id = self.memory.add_messages(
                user_id, 
                messages, 
                skip_vector_storage=not store_in_archival
            )
            
            if wait_for_completion:
                logger.info(f"Waiting for run {run_id} to complete...")
                self.memory.wait_for_run(run_id)
                logger.info(f"Run {run_id} completed")
            
            return {
                "success": True, 
                "run_id": run_id,
                "message": f"Processed {len(messages)} messages",
                "messages_count": len(messages)
            }
        except Exception as e:
            logger.error(f"Error adding conversation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_user_context(
        self, 
        org_id: str,
        customer_phone: str,
        format: str = "xml"
    ) -> Dict[str, Any]:
        """Get user memory block"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            prompt_formatted = (format == "xml")
            context = self.memory.get_user_memory(user_id, prompt_formatted=prompt_formatted)
            
            if context is None:
                return {
                    "success": False,
                    "message": f"No memory found for {user_id}",
                    "context": ""
                }
            
            return {
                "success": True,
                "context": context
            }
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": ""
            }
    
    def get_summary(
        self, 
        org_id: str,
        customer_phone: str,
        format: str = "xml"
    ) -> Dict[str, Any]:
        """Get conversation summary"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            prompt_formatted = (format == "xml")
            summary = self.memory.get_summary(user_id, prompt_formatted=prompt_formatted)
            
            if summary is None:
                return {
                    "success": False,
                    "message": f"No summary found for {user_id}",
                    "summary": ""
                }
            
            return {
                "success": True,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": ""
            }
    
    def search_memories(
        self, 
        org_id: str,
        customer_phone: str,
        query: str,
        max_results: int = 5,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Semantic search over conversation history"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            results = self.memory.search(user_id, query, tags=tags)
            limited_results = results[:max_results]
            
            logger.info(f"Found {len(results)} results for query: {query}")
            
            return {
                "success": True,
                "results": limited_results,
                "count": len(limited_results),
                "total_found": len(results)
            }
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    def get_full_context(
        self, 
        org_id: str,
        customer_phone: str,
        current_query: Optional[str] = None,
        max_search_results: int = 3,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive context including blocks and relevant memories"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            
            # Get user context block
            user_context_result = self.get_user_context(org_id, customer_phone, format="xml")
            user_context = user_context_result.get("context", "")
            
            # Get summary if requested
            summary = ""
            if include_summary:
                try:
                    summary_result = self.get_summary(org_id, customer_phone, format="xml")
                    summary = summary_result.get("summary", "")
                except Exception as e:
                    logger.error(f"Error getting summary: {e}")
                    summary = ""
            
            # Search for relevant memories if query provided
            relevant_memories = []
            if current_query:
                search_result = self.search_memories(
                    org_id,
                    customer_phone,
                    query=current_query,
                    max_results=max_search_results
                )
                relevant_memories = search_result.get("results", [])
            
            # Format combined context
            combined = self._format_combined_context(
                user_context, 
                summary,
                relevant_memories
            )
            
            return {
                "success": True,
                "user_context": user_context,
                "summary": summary,
                "relevant_memories": relevant_memories,
                "combined_context": combined
            }
        except Exception as e:
            logger.error(f"Error getting full context: {e}")
            return {
                "success": False,
                "error": str(e),
                "combined_context": ""
            }
    
    def _format_combined_context(
        self, 
        user_context: str,
        summary: str,
        memories: List[str]
    ) -> str:
        """Format blocks and memories for injection into system prompt"""
        context_parts = []
        
        if user_context:
            context_parts.append(user_context)
        
        if summary:
            context_parts.append(summary)
        
        if memories:
            memories_section = "<relevant_memories>"
            for i, mem in enumerate(memories, 1):
                memories_section += f"\n{i}. {mem}"
            memories_section += "\n</relevant_memories>"
            context_parts.append(memories_section)
        
        return "\n\n".join(context_parts)
    
    # ==================== NEW: Tag Management Methods ====================
    
    def list_org_agents(
        self, 
        org_id: str,
        limit: int = 50,
        additional_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        List all sleeptime agents for an organization
        
        Args:
            org_id: Organization identifier
            limit: Maximum number of agents to return
            additional_tags: Optional additional tag filters
        """
        try:
            tags = [f"org:{org_id}"]
            if additional_tags:
                tags.extend(additional_tags)
            
            agents = self.memory.letta_client.agents.list(
                tags=tags,
                tags_match_all=False,  # Match agents with org tag
                limit=limit
            )
            
            return {
                "success": True,
                "org_id": org_id,
                "total": len(agents),
                "agents": [
                    {
                        "agent_id": a.id,
                        "name": a.name,
                        "tags": a.tags,
                        "created_at": str(a.created_at) if hasattr(a, 'created_at') else None
                    }
                    for a in agents
                ]
            }
        except Exception as e:
            logger.error(f"Error listing org agents: {e}")
            return {
                "success": False,
                "error": str(e),
                "agents": []
            }
    
    def add_tags(
        self,
        org_id: str,
        customer_phone: str,
        new_tags: List[str]
    ) -> Dict[str, Any]:
        """
        Add tags to an existing customer's sleeptime agent
        
        Args:
            org_id: Organization identifier
            customer_phone: Customer phone number
            new_tags: List of tags to add
        """
        try:
            user_id = f"{org_id}:{customer_phone}"
            agent_id = self.memory.get_memory_agent_id(user_id)
            
            if not agent_id:
                return {
                    "success": False,
                    "error": f"No agent found for {user_id}"
                }
            
            agent = self.memory.letta_client.agents.retrieve(agent_id)
            updated_tags = list(set(agent.tags + new_tags))
            
            self.memory.letta_client.agents.modify(
                agent_id=agent_id,
                tags=updated_tags
            )
            
            logger.info(f"Added tags {new_tags} to agent {agent_id}")
            return {
                "success": True,
                "agent_id": agent_id,
                "tags": updated_tags,
                "added": new_tags
            }
        except Exception as e:
            logger.error(f"Error adding tags: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def remove_tags(
        self,
        org_id: str,
        customer_phone: str,
        tags_to_remove: List[str]
    ) -> Dict[str, Any]:
        """
        Remove tags from a customer's sleeptime agent
        
        Args:
            org_id: Organization identifier
            customer_phone: Customer phone number
            tags_to_remove: List of tags to remove
        """
        try:
            user_id = f"{org_id}:{customer_phone}"
            agent_id = self.memory.get_memory_agent_id(user_id)
            
            if not agent_id:
                return {
                    "success": False,
                    "error": f"No agent found for {user_id}"
                }
            
            agent = self.memory.letta_client.agents.retrieve(agent_id)
            
            # Don't allow removing critical tags
            protected_tags = ["ai-memory-sdk", f"org:{org_id}", f"customer:{customer_phone}", "type:memory-agent"]
            removable_tags = [t for t in tags_to_remove if t not in protected_tags]
            
            updated_tags = [t for t in agent.tags if t not in removable_tags]
            
            self.memory.letta_client.agents.modify(
                agent_id=agent_id,
                tags=updated_tags
            )
            
            logger.info(f"Removed tags {removable_tags} from agent {agent_id}")
            return {
                "success": True,
                "agent_id": agent_id,
                "tags": updated_tags,
                "removed": removable_tags,
                "protected": [t for t in tags_to_remove if t in protected_tags]
            }
        except Exception as e:
            logger.error(f"Error removing tags: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_agent_tags(
        self,
        org_id: str,
        customer_phone: str
    ) -> Dict[str, Any]:
        """Get all tags for a customer's sleeptime agent"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            agent_id = self.memory.get_memory_agent_id(user_id)
            
            if not agent_id:
                return {
                    "success": False,
                    "error": f"No agent found for {user_id}"
                }
            
            agent = self.memory.letta_client.agents.retrieve(agent_id)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "tags": agent.tags
            }
        except Exception as e:
            logger.error(f"Error getting tags: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_org_stats(self, org_id: str) -> Dict[str, Any]:
        """Get statistics for an organization"""
        try:
            agents_result = self.list_org_agents(org_id, limit=1000)
            
            if not agents_result["success"]:
                return agents_result
            
            agents = agents_result["agents"]
            
            # Analyze tags
            tag_counts = {}
            for agent in agents:
                for tag in agent["tags"]:
                    # Skip system tags for stats
                    if not tag.startswith("org:") and not tag.startswith("customer:") and tag not in ["ai-memory-sdk", "type:memory-agent"]:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return {
                "success": True,
                "org_id": org_id,
                "total_agents": len(agents),
                "tag_distribution": tag_counts
            }
        except Exception as e:
            logger.error(f"Error getting org stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== Existing Methods (keeping for backward compatibility) ====================
    
    def delete_user(self, org_id: str, customer_phone: str) -> Dict[str, Any]:
        """Delete all memory for a customer"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            self.memory.delete_user(user_id)
            logger.info(f"Deleted memory for {user_id}")
            return {
                "success": True,
                "message": f"Deleted all memory for {user_id}"
            }
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_agent_id(self, org_id: str, customer_phone: str) -> Dict[str, Any]:
        """Get the Letta sleeptime agent ID for a customer"""
        try:
            user_id = f"{org_id}:{customer_phone}"
            agent_id = self.memory.get_memory_agent_id(user_id)
            
            if agent_id is None:
                return {
                    "success": False,
                    "message": f"No agent found for {user_id}"
                }
            
            return {
                "success": True,
                "agent_id": agent_id,
                "dashboard_url": f"https://app.letta.com/agents/{agent_id}"
            }
        except Exception as e:
            logger.error(f"Error getting agent ID: {e}")
            return {
                "success": False,
                "error": str(e)
            }