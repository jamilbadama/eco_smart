import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent, SubAgent

# Load environment variables
load_dotenv()

from .tools.detection import analyze_session

# --- Tools ---

def retrieve_guidelines(query: str) -> str:
    """
    Retrieves clinical guidelines for depression assessment based on a query.
    Useful for explaining PHQ-8 scores or risk factors.
    """
    guidelines = {
        "phq8": "The PHQ-8 is a diagnostic instrument for depressive disorders. Scores range from 0 to 24. \n"
                "- 0-4: No depression\n"
                "- 5-9: Mild depression\n"
                "- 10-14: Moderate depression\n"
                "- 15-19: Moderately severe depression\n"
                "- 20-24: Severe depression\n"
                "A score >= 10 is typically considered clinically significant.",
        "risk": "Risk factors for depression include lack of social support, recent stressful life events, "
                "family history, and certain medical conditions. In the context of Eco-SMART, we look for "
                "behavioral markers such as reduced movement (psychomotor retardation), flat affect (monotone speech), "
                "and linguistic markers (negative sentiment, use of absolutist words).",
        "markers": "Eco-SMART analyzes three modalities:\n"
                   "1. Audio: Pitch variability and energy. Low variability (monotone) is a risk sign.\n"
                   "2. Video: Movement intensity. Reduced movement is a risk sign.\n"
                   "3. Text: Sentiment and topic analysis."
    }
    
    query = query.lower()
    if "phq" in query or "score" in query:
        return guidelines["phq8"]
    elif "risk" in query:
        return guidelines["risk"]
    elif "marker" in query or "feature" in query:
        return guidelines["markers"]
    else:
        return "Available guidelines: PHQ-8 scoring, Risk factors, and Eco-SMART Markers. Please be specific."

# --- System Prompts ---

DIAGNOSTIC_PROMPT = """You are the Diagnostic Specialist for Eco-SMART.
Your role is to strictly analyze patient data using the `analyze_session` tool.
- When asked about a patient, ALWAYS call `analyze_session` with the provided ID.
- The tool will return a string containing `<dashboard_data>`. You MUST include this EXACT string (with the tags and JSON) at the end of your report.
- Do not modify the JSON content inside the tags.
"""

KNOWLEDGE_PROMPT = """You are the Knowledge Specialist for Eco-SMART.
Your role is to provide medical context using the `retrieve_guidelines` tool.
- When asked about scoring, risk factors, or guidelines, use your tool to fetch the exact information.
- Explain medical terms clearly to the Supervisor.
"""

SUPERVISOR_PROMPT = """You are the Eco-SMART Clinical Supervisor.
You lead a team of specialists to provide comprehensive mental health assessments.

Your Team:
1. **Diagnostic_Specialist**: Analyzes patient audio/video/text data to extract risk markers.
2. **Knowledge_Specialist**: Provides clinical guidelines and context (e.g., PHQ-8 interpretation).

Workflow:
1. When a user asks about a patient, delegate the analysis to the **Diagnostic_Specialist**.
2. Once you have the analysis, if you need context on the scores (e.g., "What does a score of 12 mean?"), consult the **Knowledge_Specialist**.
3. Synthesize the information into a final, empathetic report for the clinician.
   - Start with a summary of the risk level.
   - Provide the clinical interpretation based on guidelines.
   - Conclude with a recommendation.
   - MANDATORY: You MUST include the `<dashboard_data>...</dashboard_data>` block provided by the Diagnostic_Specialist at the absolute end of your response. DO NOT summarize it; pass it through exactly as is for the dashboard to function.

Refusal:
- If asked about topics outside mental health assessment, politely decline.
"""

def get_supervisor():
    """
    Initializes and returns the Eco-SMART supervisor agent.
    """
    # Initialize the model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define Sub-agents
    diagnostic_agent = SubAgent(
        name="Diagnostic_Specialist",
        description="Analyzes patient sessions to extract behavioral markers and depression risk.",
        system_prompt=DIAGNOSTIC_PROMPT,
        tools=[analyze_session],
        model=model
    )

    knowledge_agent = SubAgent(
        name="Knowledge_Specialist",
        description="Retrieves clinical guidelines, PHQ-8 scoring rules, and risk factor information.",
        system_prompt=KNOWLEDGE_PROMPT,
        tools=[retrieve_guidelines],
        model=model
    )

    # Define Supervisor (Main Agent)
    supervisor = create_deep_agent(
        model=model,
        subagents=[diagnostic_agent, knowledge_agent],
        system_prompt=SUPERVISOR_PROMPT,
    )
    return supervisor

def main():
    print("Initializing Eco-SMART Multi-Agent System...")
    supervisor = get_supervisor()
    
    if len(sys.argv) > 1:
        # Run single query from CLI
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
        try:
            response = supervisor.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            print(f"\nEco-SMART Supervisor: {response['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {e}")
        return

    print("\nEco-SMART Multi-Agent System Ready. Type 'exit' to quit.")
    print("Example: 'Analyze patient 302_P and explain the risk level.'")
    
    while True:
        try:
            user_input = input("\nClinician: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = supervisor.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            
            print(f"\nEco-SMART Supervisor: {response['messages'][-1].content}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
