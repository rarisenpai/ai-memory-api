from re import I
from openai import OpenAI
from ai_memory_sdk import Memory

openai_client = OpenAI()
memory = Memory()

max_messages = 10
keep_last_n_messages = 3

def compact(messages, user_id: str = "default_user") -> str:
    """ Compact the messages to keep only the last n messages """

    # learn message state 
    run = memory.add_messages(user_id, messages)
    messages = messages[-keep_last_n_messages:]

    # get the user memory and summary (can also run in another thread)
    memory.wait_for_run(run)

    # print the summary 
    summary = memory.get_summary(user_id, prompt_formatted=True)
    user_memory = memory.get_user_memory(user_id, prompt_formatted=True)
    print(f"Summary: {summary}")
    print(f"User Memory: {user_memory}")

    return messages 

def chat_with_memories(message: str, message_history, user_id: str = "default_user") -> str:

    # get the user memory 
    user_memory = memory.get_user_memory(user_id)
    if not user_memory:
        memory.initialize_user_memory(user_id, reset=True)
        user_memory = memory.get_user_memory(user_id)
    
    # format the user memory 
    user_memory_prompt= memory.get_user_memory(user_id, prompt_formatted=True)
    summary_prompt = memory.get_summary(user_id, prompt_formatted=True)

    # generate the assistant response
    system_prompt = f"<system>You are a helpful AI assistant</system>"
    system_prompt += f"\n{user_memory_prompt}"
    system_prompt += f"\n{summary_prompt}"
    messages = [{"role": "system", "content": system_prompt}] + message_history + [{"role": "user", "content": message}]
    response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    assistant_response = response.choices[0].message.content

    # Create new memories from the conversation
    message_history.append({"role": "user", "content": message})
    message_history.append({"role": "assistant", "content": assistant_response})

    # compact the messages and update the memory/summary
    print(f"Messages len is {len(message_history)} / {max_messages}")
    if len(message_history) > max_messages:
        print(f"Messages len is {len(message_history)}, compacting...")
        message_history = compact(message_history, user_id)
        print(f"Messages len is {len(message_history)}, compacted")

    return assistant_response, message_history

def main():
    print("Chat with AI (type 'exit' to quit)")
    message_history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        assistant_response, message_history = chat_with_memories(user_input, message_history)
        print(f"AI: {assistant_response}")

if __name__ == "__main__":
    main()