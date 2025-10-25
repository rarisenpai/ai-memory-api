import sys
import types

# Mock letta_client module
letta_client = types.ModuleType("letta_client")

class Letta:
    def __init__(self, token=None, base_url=None):
        self.token = token
        self.base_url = base_url

letta_client.Letta = Letta
sys.modules['letta_client'] = letta_client


from ai_memory_sdk import Memory  # noqa: E402

def test_self_hosted_initialization_with_token():
    """Test initialization for self-hosted Letta with authentication"""
    memory = Memory(
        base_url="http://localhost:8283",
        api_key="my_token"
    )

    
    assert memory.letta_client.base_url == "http://localhost:8283"
    assert memory.letta_client.token == "my_token"

