import pytest
from ai_memory_sdk import Memory 
import os
import time

@pytest.fixture(scope="module")
def client():
    return Memory(api_key=os.getenv("LETTA_API_KEY"))

# create fixture for user 
@pytest.fixture(scope="module")
def user_id(client):
    user_id = "test_user_id_123"
    client.initialize_user_memory(user_id, reset=True)
    yield user_id
    client.delete_user(user_id)


def test_conversational_memory(user_id, client):
    """ Test the conversational memory client """

    # add messages 
    run =  client.add_messages(user_id, messages=[
        {
            "role": "user",
            "content": "Hi my name is Bob"
        }, 
        {
            "role": "assistant",
            "content": "Hi Bob, how can I help you today?"
        },
        {
            "role": "user",
            "content": "I love cats"
        }
    ], skip_vector_storage=False)

    # wait for the run to complete
    client.wait_for_run(run)

    # get the memory 
    memory = client.get_user_memory(user_id)
    assert "Bob" in memory, f"Expected 'Bob' in memory value, got {memory}"
    print(memory)

    # search 
    search_results = client.search(user_id, "animals")
    assert len(search_results) > 0, f"Expected search results, got {search_results}"
    print("SEARCH RESULTS", search_results)
    assert "cats" in search_results[0], f"Expected 'cats' in search results, got {search_results[0]}"
    print(search_results)


