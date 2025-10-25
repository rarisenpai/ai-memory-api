from typing import List, Dict, Any
from schemas import Message, File


messages_prompt = "The following message interactions have occured"
messages_tag = "messages"
message_total_char_limit = 5000 # how many character are included in a message chunk accross all messages

file_tag = "file"
file_part_tag = "file_part"
file_char_limit = 20000 # how many character to include in file part 

def format_messages(messages: List[Message]) -> List[str]:
    """
    Format messages like the following: 

    user: hi my name is sarah
    assistant: Hello Sarah! I'm Sam. smiles warmly There's something special about first meetings, don't you think? Like opening a book to its first page, full of possibilities. I'd love to get to know you better - what brings you here today?
    assistant: Tool call returned Sent message successfully.

    """
    message_history = "\n".join([
        f"{msg.role}: {msg.content}" for msg in messages
    ])

    return [{"role": "user", "content": f"<{messages_tag}>{messages_prompt}:\n{message_history}</{messages_tag}>"}]

def format_files(files: List[File]) -> List[Dict[str, str]]:
    """
    Format files into multiple separate messages, ensuring each message stays under file_char_limit.
    Each file part becomes its own message.
    """
    all_messages = []
    
    for file in files:
        try:
            # Read file content from disk
            with open(file.file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            # If we can't read the file, send an error message
            error_msg = f"<{file_tag} label=\"{file.label}\" description=\"{file.description}\">[Error reading file: {str(e)}]</{file_tag}>"
            all_messages.append({"role": "user", "content": error_msg})
            continue
            
        # chunk up file content into parts 
        file_content_chunks = [file_content[i:i + file_char_limit] for i in range(0, len(file_content), file_char_limit)]
        
        # Create a separate message for each file part
        for i, chunk in enumerate(file_content_chunks):
            part_number = i + 1
            total_parts = len(file_content_chunks)
            
            # Create the message content for this specific part
            file_part_content = f"<{file_part_tag} part={part_number}/{total_parts}>{chunk}</{file_part_tag}>"
            file_message = f"<{file_tag} label=\"{file.label}\" description=\"{file.description}\">{file_part_content}</{file_tag}>"
            
            all_messages.append({"role": "user", "content": file_message})
    
    return all_messages
    

    
