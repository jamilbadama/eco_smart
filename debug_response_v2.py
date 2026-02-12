from src.eco_smart_multi_agent import get_supervisor
import json

supervisor = get_supervisor()
query = "Analyze patient 302_P and explain the risk level."

print("Invoking supervisor...")
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": query}]}
)

print("\n--- MESSAGE STRUCTURE ---")
for i, msg in enumerate(response['messages']):
    name = getattr(msg, 'name', 'N/A')
    role = getattr(msg, 'role', 'N/A')
    msg_type = type(msg).__name__
    
    # Check for ToolMessage specific fields
    tool_id = getattr(msg, 'tool_call_id', 'N/A')
    
    # Check for AICall (if deepagents uses it)
    tool_calls = getattr(msg, 'tool_calls', 'N/A')

    print(f"Index: {i}")
    print(f"  Type: {msg_type}")
    print(f"  Role: {role}")
    print(f"  Name: {name}")
    print(f"  Tool ID: {tool_id}")
    print(f"  Tool Calls: {tool_calls}")
    
    content_str = str(msg.content)
    if '"session_id"' in content_str:
        print(f"  [!] CONTAINS CLINICAL DATA")
        # print(content_str[:500]) # Don't print too much to avoid truncation
    
    print("-" * 20)

print("\nTotal messages:", len(response['messages']))
