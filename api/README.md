# Memory Layer API Reference

The Memory Layer API provides a high-level RESTful interface on top of the `ai-memory-sdk`. It's designed to be used as a persistent, stateful memory store by external systems like **n8n**, agent frameworks, or other applications without them needing to integrate the Python SDK directly.

This service exposes Letta's stateful agent architecture as a simple memory service. You interact with it by:

1.  **Initializing** a memory store for a specific `user_id` (which corresponds to a "subject" or Letta agent).
2.  **Defining Blocks** (e.g., `human`, `policies`) that the agent will manage.
3.  **Adding Conversations** (messages) that the agent processes to automatically update its blocks.
4.  **Retrieving Context** (the blocks and relevant memories) to inject into your own agent's system prompt.

-----

## Setup & Running

To run this service, you need to run the included FastAPI server.

### 1\. Install Dependencies

You will need the `ai-memory-sdk`, `uv` (as a fast installer/runner), and `fastapi`.

**Standard Installation (as a user):**

**Local Development (from source):**
This assumes you are developing the `ai-memory-sdk` and the API service from the same repository.

```bash
# Navigate to the Python source directory
cd src/python/

# Install the ai-memory-sdk in editable (-e) mode using uv
uv pip install -e .

# Also install FastAPI in the same environment
uv pip install "fastapi[all]"
```

### 2\. Set Environment Variables

The service is configured using environment variables. The most important is `LETTA_API_KEY`.

```bash
# Required for Letta Cloud
export LETTA_API_KEY="your_letta_api_key_here"

# Optional: Override default models used by the service
export LETTA_MODEL="openai/gpt-4o-mini"
export LETTA_EMBEDDING="openai/text-embedding-3-small"

# Optional: For self-hosted Letta
export LETTA_BASE_URL="http://localhost:8283"
```

### 3\. Run the Server

From the `src/python/` directory, run the application using `uv run`:

```bash
# From the src/python/ directory
cd src/python/
uv run uvicorn api.main:app --reload --port 3000
```

  * This command tells `uvicorn` to run the `app` object from the `api.main` module.
  * `--reload` automatically restarts the server when you make code changes.
  * `--port 3000` sets the server port. You can change this to any port you prefer.

Once running, the API documentation will be available at `http://localhost:3000/docs`.

-----

## API Endpoints

All endpoints are prefixed with `/memory`.

### Agent & Block Initialization

These endpoints create or update the Letta agent associated with a `user_id`.

#### `POST /memory/initialize`

Initialize memory for a user with the default blocks: `human` and `summary`. This is a simplified helper for basic chat use cases.

**Request Body (`InitializeUserRequest`)**

```json
{
  "user_id": "user_456",
  "user_info": "This is Sarah. She is a new customer.",
  "reset": false
}
```

  * `reset: true` will delete and recreate the agent if it already exists.

**Success Response (200) (`InitializeUserResponse`)**

```json
{
  "success": true,
  "message": "Initialized memory for user_456",
  "agent_id": "agent_abc123..."
}
```

-----

#### `POST /memory/initialize-with-blocks`

Initialize memory for a user with a **custom set of blocks**. This is the recommended endpoint for advanced use cases (e.g., support bots, personal assistants).

The service will **always ensure** a `human` and `summary` block exist, even if not provided.

**Request Body (`InitializeWithBlocksRequest`)**

```json
{
  "user_id": "customer_123",
  "blocks": [
    {
      "label": "customer_profile",
      "description": "Basic customer info and support tier.",
      "value": "Premium subscriber. Joined 2024.",
      "char_limit": 5000
    },
    {
      "label": "policies",
      "description": "Company policies relevant to this customer.",
      "value": "Refund policy: 30 days. Premium support: 24/7.",
      "char_limit": 10000
    }
  ],
  "reset": false
}
```

**Success Response (200) (`InitializeWithBlocksResponse`)**

```json
{
  "success": true,
  "message": "Initialized memory for customer_123",
  "agent_id": "agent_xyz789...",
  "blocks_created": 4
}
```

  * `blocks_created` includes your custom blocks plus the default `human` and `summary` blocks.

-----

### Adding & Processing Memory

#### `POST /memory/add`

Add conversation messages to memory. The underlying Letta agent will process these messages asynchronously and update its memory blocks based on their content and the block descriptions.

**Request Body (`AddConversationRequest`)**

```json
{
  "user_id": "customer_123",
  "messages": [
    { "role": "user", "content": "Hi, I need to return an item." },
    { "role": "assistant", "content": "I can help with that. What is the order number?" },
    { "role": "user", "content": "It's 12345-ABC." }
  ],
  "store_in_archival": true,
  "wait_for_completion": true
}
```

  * `store_in_archival: true`: (Default) Saves messages to Letta's archival (vector) memory, enabling semantic search via `/search`. Set to `false` to save costs if you don't need search.
  * `wait_for_completion: true`: (Default) The API call will wait until the Letta agent has finished processing the messages and updating its blocks. Set to `false` for a "fire-and-forget" call that returns immediately.

**Success Response (200) (`AddConversationResponse`)**

```json
{
  "success": true,
  "run_id": "run_123...",
  "message": "Processed 3 messages",
  "messages_count": 3
}
```

-----

### Retrieving Context

These endpoints retrieve the agent's memory for use in a system prompt.

#### `GET /memory/context`

Get the comprehensive, prompt-ready context for a user. This is the **primary endpoint** for retrieving memory.

It combines all user-defined blocks (like `customer_profile`), the default blocks (`human`, `summary`), and (if a `query` is provided) semantically relevant memories from the archive.

**Query Parameters**

  * `user_id` (required): The user identifier.
  * `query` (optional): The *current* user message. Used to find relevant memories.
  * `max_results` (optional): Number of search results to include. Default: 3.
  * `include_summary` (optional): Whether to include the `summary` block. Default: true.

**Example Request**
`GET /memory/context?user_id=customer_123&query=What's your refund policy?&max_results=2`

**Success Response (200) (`FullContextResponse`)**

```json
{
  "success": true,
  "user_context": "<human description=\"Information about the human user.\">Premium subscriber...</human>",
  "summary": "<summary description=\"A rolling summary of the conversation.\">User asked to return order 12345-ABC.</summary>",
  "relevant_memories": [
    "User: Hi, I need to return an item."
  ],
  "combined_context": "<human description=\"Information about the human user.\">Premium subscriber...</human>\n\n<summary description=\"A rolling summary of the conversation.\">User asked to return order 12345-ABC.</summary>\n\n<relevant_memories>\n1. User: Hi, I need to return an item.\n</relevant_memories>"
}
```

  * **`combined_context`** is the final, formatted string ready to be injected into your LLM's system prompt. (Note: This example only shows default blocks; custom blocks like `customer_profile` would also be included here).

-----

#### `GET /memory/user-context`

A helper to get *only* the `human` block content.

**Query Parameters**

  * `user_id` (required): The user identifier.
  * `format` (optional): `'xml'` (default) or `'raw'` (for just the text value).

**Example Request**
`GET /memory/user-context?user_id=customer_123&format=raw`

**Success Response (200) (`ContextResponse`)**

```json
{
  "success": true,
  "context": "Premium subscriber. Joined 2024."
}
```

-----

#### `GET /memory/summary`

A helper to get *only* the `summary` block content.

**Query Parameters**

  * `user_id` (required): The user identifier.
  * `format` (optional): `'xml'` (default) or `'raw'`.

**Example Request**
`GET /memory/summary?user_id=customer_123`

**Success Response (200) (`SummaryResponse`)**

```json
{
  "success": true,
  "summary": "<summary description=\"A rolling summary...\">User asked to return order 12345-ABC.</summary>"
}
```

-----

### Searching Archival Memory

#### `GET /memory/search`

Perform a semantic search over a user's conversation history. This requires `store_in_archival: true` to have been used during `POST /add`.

**Query Parameters**

  * `user_id` (required): The user identifier.
  * `query` (required): The search query.
  * `max_results` (optional): Max results to return. Default: 5.
  * `tags` (optional): Comma-separated tags to filter by (e.g., `user`, `assistant`).

**Example Request**
`GET /memory/search?user_id=customer_123&query=order number&tags=user`

**Success Response (200) (`SearchResult`)**

```json
{
  "success": true,
  "results": [
    "It's 12345-ABC."
  ],
  "count": 1,
  "total_found": 1
}
```

-----

### Management & Debugging

#### `DELETE /memory/user/{user_id}`

Delete all memory for a user. This **permanently deletes** the Letta agent and all its associated data (blocks and archival memory).

**Example Request**
`DELETE /memory/user/customer_123`

**Success Response (200) (`DeleteResponse`)**

```json
{
  "success": true,
  "message": "Deleted all memory for user customer_123"
}
```

-----

#### `GET /memory/agent/{user_id}`

Get the underlying Letta agent ID and a direct link to view its memory and configuration in the Letta dashboard. This is extremely useful for debugging.

**Example Request**
`GET /memory/agent/customer_123`

**Success Response (200) (`AgentIdResponse`)**

```json
{
  "success": true,
  "agent_id": "agent_xyz789...",
  "dashboard_url": "https://app.letta.com/agents/agent_xyz789..."
}
```

-----

## Full Example Workflow (cURL)

Here is a full end-to-end example of using the API.

**1. Initialize a new user with custom blocks**

```bash
curl -X 'POST' \
  'http://localhost:3000/memory/initialize-with-blocks' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "n8n_user_007",
    "blocks": [
      {
        "label": "user_profile",
        "description": "Information about the user, their role, and preferences.",
        "value": "User is a marketing manager named Alex."
      },
      {
        "label": "current_project",
        "description": "Details about the project Alex is currently working on.",
        "value": "Project: Q4 product launch."
      }
    ],
    "reset": true
  }'
```

**2. Add a conversation**

```bash
curl -X 'POST' \
  'http://localhost:3000/memory/add' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "n8n_user_007",
    "messages": [
      { "role": "user", "content": "What's the main goal for the Q4 launch?" },
      { "role": "assistant", "content": "The main goal is to increase user acquisition by 15%." },
      { "role": "user", "content": "Got it. And what's my personal preference for email tone?" },
      { "role": "assistant", "content": "Your profile notes you prefer a professional but friendly tone." }
    ],
    "store_in_archival": true,
    "wait_for_completion": true
  }'
```

**3. Get the full context (simulating a new user message)**
This call will retrieve the blocks (which the agent updated) and search for relevant past messages.

```bash
curl -X 'GET' \
  'http://localhost:3000/memory/context?user_id=n8n_user_007&query=remind%20me%20about%20the%20launch%20goal' \
  -H 'Accept: application/json'
```

**Expected Response:**
The `combined_context` will contain the `user_profile` block, the `current_project` block (now updated by the agent), the `summary`, and the relevant memory about the "15% user acquisition" goal. This string is ready to be passed to an LLM.
