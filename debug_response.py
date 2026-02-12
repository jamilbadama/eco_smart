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
    print(f"[{i}] Type: {msg_type}, Role: {role}, Name: {name}")
    content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
    print(f"    Content: {content_preview}")
    
    # Check for specific attributes that deepagents might use
    # Tool outputs in LangChain are often in ToolMessage
    if hasattr(msg, 'tool_call_id'):
         print(f"    Tool Call ID: {msg.tool_call_id}")

print("\n--- FINAL MESSAGE CONTENT ---")
print(response['messages'][-1].content)
