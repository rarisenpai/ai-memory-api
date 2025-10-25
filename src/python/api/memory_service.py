from ai_memory_sdk import Memory
from typing import List, Dict, Optional, Any
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from ai_memory_sdk import Memory
from typing import List, Dict, Optional, Any
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryService:
    """
    High-level memory service for n8n integration
    
    Uses tags for user identification (no Letta identities)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        subject_id: Optional[str] = None,
        model: str = "openai/gpt-4.1",
        embedding: str = "openai/text-embedding-3-small"
    ):
        """
        Initialize Memory Service
        
        Args:
            api_key: Letta API key (or from LETTA_API_KEY env var)
            base_url: Letta server URL (or from LETTA_BASE_URL env var)
            subject_id: Optional subject binding for single-user mode
            model: LLM model to use (default: openai/gpt-4.1)
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
    
    def initialize_user(
        self, 
        user_id: Optional[str] = None,
        user_info: str = "",
        reset: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize memory for a user with default blocks (human, summary)
        """
        try:
            effective_user_id = self._get_user_id(user_id)
            
            # Check if agent already exists
            agent = self.memory._get_matching_agent(tags=[effective_user_id])
            
            if agent:
                if reset:
                    self.memory._delete_agent(agent.id)
                    agent = None
                else:
                    return {
                        "success": True,
                        "message": f"Memory already exists for {effective_user_id}",
                        "agent_id": agent.id
                    }
            
            # Create agent with embedding
            agent_id = self.memory.letta_client.agents.create(
                name=f"memory_agent_{effective_user_id}",
                model=self.model,
                embedding=self.embedding,
                agent_type="sleeptime_agent",
                tags=self.memory._subject_tags(effective_user_id),
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
                text=f"Initialized memory for user {effective_user_id}",
                tags=[self.memory._default_tag]
            )
            
            logger.info(f"Initialized memory for user {effective_user_id}")
            return {
                "success": True, 
                "message": f"Initialized memory for {effective_user_id}",
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error initializing user: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def initialize_with_blocks(
            self,
            user_id: Optional[str] = None,
            blocks: Optional[List[Dict[str, Any]]] = None,
            reset: bool = False
        ) -> Dict[str, Any]:
            """
            Initialize memory with custom blocks
            """
            try:
                effective_user_id = self._get_user_id(user_id)
                
                # Check if agent exists
                agent = self.memory._get_matching_agent(tags=[effective_user_id])
                
                if agent:
                    if reset:
                        self.memory._delete_agent(agent.id)
                        agent = None
                    else:
                        return {
                            "success": False,
                            "error": f"Memory already exists for {effective_user_id}. Use reset=True to recreate."
                        }
                        
                # if human, summary are not added to the blocks argument by user, add them first check whether they are already added if not add them
                
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
                if blocks:
                    for block in blocks:
                        memory_blocks.append({
                            "label": block.get("label"),
                            "description": block.get("description"),
                            "value": block.get("value", ""),
                            "limit": block.get("char_limit", 10000)
                        })
                
                # Create agent with embedding
                agent_id = self.memory.letta_client.agents.create(
                    name=f"memory_agent_{effective_user_id}",
                    model=self.model,
                    embedding=self.embedding,
                    agent_type="sleeptime_agent",
                    tags=self.memory._subject_tags(effective_user_id),
                    memory_blocks=memory_blocks
                ).id
                
                # Create initial passage
                self.memory.letta_client.agents.passages.create(
                    agent_id=agent_id,
                    text=f"Initialized memory for user {effective_user_id}",
                    tags=[self.memory._default_tag]
                )
                
                logger.info(f"Created agent with {len(memory_blocks)} blocks")
                return {
                    "success": True,
                    "message": f"Initialized memory for {effective_user_id}",
                    "agent_id": agent_id,
                    "blocks_created": len(memory_blocks)
                }
                
            except Exception as e:
                logger.error(f"Error initializing with blocks: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
    def add_conversation(
        self, 
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        store_in_archival: bool = True,
        wait_for_completion: bool = True
    ) -> Dict[str, Any]:
        """Add conversation messages to memory"""
        try:
            effective_user_id = self._get_user_id(user_id)
            
            # Ensure user exists
            self.initialize_user(effective_user_id)
            
            logger.info(f"Adding {len(messages)} messages for user {effective_user_id}")
            run_id = self.memory.add_messages(
                effective_user_id, 
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
        user_id: Optional[str] = None,
        format: str = "xml"
    ) -> Dict[str, Any]:
        """Get user memory block"""
        try:
            effective_user_id = self._get_user_id(user_id)
            prompt_formatted = (format == "xml")
            context = self.memory.get_user_memory(effective_user_id, prompt_formatted=prompt_formatted)
            
            if context is None:
                return {
                    "success": False,
                    "message": f"No memory found for user {effective_user_id}",
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
        user_id: Optional[str] = None,
        format: str = "xml"
    ) -> Dict[str, Any]:
        """Get conversation summary"""
        try:
            effective_user_id = self._get_user_id(user_id)
            prompt_formatted = (format == "xml")
            summary = self.memory.get_summary(effective_user_id, prompt_formatted=prompt_formatted)
            
            if summary is None:
                return {
                    "success": False,
                    "message": f"No summary found for user {effective_user_id}",
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
        query: str,
        user_id: Optional[str] = None,
        max_results: int = 5,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Semantic search over conversation history"""
        try:
            effective_user_id = self._get_user_id(user_id)
            results = self.memory.search(effective_user_id, query, tags=tags)
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
        current_query: Optional[str] = None,
        user_id: Optional[str] = None,
        max_search_results: int = 3,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive context including blocks and relevant memories"""
        try:
            effective_user_id = self._get_user_id(user_id)
            
            # Get user context block
            user_context_result = self.get_user_context(effective_user_id, format="xml")
            user_context = user_context_result.get("context", "")
            
            # Get summary if requested
            summary = ""
            if include_summary:
                try:
                    summary_result = self.get_summary(effective_user_id, format="xml")
                    summary = summary_result.get("summary", "")
                except Exception as e:
                    logger.error(f"Error getting summary: {e}")
                    summary = ""
            
            # Search for relevant memories if query provided
            relevant_memories = []
            if current_query:
                search_result = self.search_memories(
                    current_query,
                    user_id=effective_user_id, 
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
    
    def delete_user(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete all memory for a user"""
        try:
            effective_user_id = self._get_user_id(user_id)
            self.memory.delete_user(effective_user_id)
            logger.info(f"Deleted memory for user {effective_user_id}")
            return {
                "success": True,
                "message": f"Deleted all memory for user {effective_user_id}"
            }
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_agent_id(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the Letta agent ID for a user"""
        try:
            effective_user_id = self._get_user_id(user_id)
            agent_id = self.memory.get_memory_agent_id(effective_user_id)
            
            if agent_id is None:
                return {
                    "success": False,
                    "message": f"No agent found for user {effective_user_id}"
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