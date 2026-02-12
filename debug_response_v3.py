from src.eco_smart_multi_agent import get_supervisor
import json

supervisor = get_supervisor()
query = "Analyze patient 302_P and explain the risk level."

print("Invoking supervisor...")
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": query}]}
)

print("\n--- TOOL MESSAGE INSPECTION ---")
for i, msg in enumerate(response['messages']):
    name = getattr(msg, 'name', 'N/A')
    if name == 'task' or type(msg).__name__ == 'ToolMessage':
        print(f"Index: {i}")
        print(f"Name: {name}")
        print("Content Type:", type(msg.content))
        print("Content Preview:", str(msg.content)[:1000])
        print("-" * 40)
