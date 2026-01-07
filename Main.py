import os
from dotenv import load_dotenv
from tavily import TavilyClient
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from state import ResearchState  # Make sure state.py is in the same folder
from pydantic import BaseModel, Field
from typing import Literal
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 

# Load environment variables
load_dotenv()

# --- 1. DEFINE NODES (The Agents) ---
class ManagerDecision(BaseModel):
    next_step: Literal["web_researcher", "pdf_analyst", "writer"] = Field(
        description="The next agent to act."
    )
    reasoning: str = Field(description="Why this step was chosen.")
    
def manager_node(state: ResearchState):
    print("--- MANAGER: Assessing State & Deciding Next Step ---")
    
    # 1. Initialize the LLM with Structured Output
    # This forces the LLM to adhere to the Pydantic schema
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ManagerDecision)
    
    # 2. Context for the Manager
    # We show it what data we currently have so it can decide what is missing.
    web_status = "Done" if state.get('web_findings') else "Not Started"
    pdf_status = "Done" if state.get('pdf_insights') else "Not Started"
    
    system_msg = SystemMessage(content="""
        You are the Orchestrator of a Research Team. 
        Your job is to decide the next step based on the User Request and current progress.
        
        Available Agents:
        1. 'web_researcher': Use this if we need external market data.
        2. 'pdf_analyst': Use this if we need internal document data.
        3. 'writer': Use this ONLY when we have sufficient data from BOTH sources (or if one source is explicitly skipped).
        
        Goal: Complete the user's request efficiently. Do not repeat steps.
    """)
    
    user_msg = HumanMessage(content=f"""
        User Request: {state['user_request']}
        
        Current Status:
        - Web Research: {web_status}
        - PDF Analysis: {pdf_status}
        
        What is the next step?
    """)
    
    # 3. Get the Decision
    decision = structured_llm.invoke([system_msg, user_msg])
    
    print(f"--- DECISION: {decision.next_step.upper()} ({decision.reasoning}) ---")
    
    # 4. Save decision to state so the Router can see it
    # Note: We need to update our State definition to hold this 'next_step'
    return {"current_step": decision.next_step}

def web_research_node(state: ResearchState):
    print("--- RESEARCHER: Searching Real Web ---")
    
    # Initialize Client
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Execute Search
    response = tavily.search(
        query=state['user_request'], 
        search_depth="advanced", 
        max_results=3
    )
    
    # Parse Results
    web_results = []
    for result in response['results']:
        web_results.append({
            "url": result['url'],
            "title": result['title'],
            "snippet": result['content'][:200] + "...", 
            "credibility_score": 8 
        })
        
    print(f"--- Found {len(web_results)} articles ---")
    return {"web_findings": web_results}

def pdf_analyst_node(state: ResearchState):
    print("--- ANALYST: Checking Internal Knowledge Base ---")
    
    pdf_path = "source_doc.pdf"
    index_folder = "faiss_index"  # Folder to save the math
    
    if not os.path.exists(pdf_path):
        return {"pdf_insights": []}

    # Initialize Embeddings (The Math Engine)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # --- CACHING LOGIC ---
    # Scenario A: We already did the work. Load it.
    if os.path.exists(index_folder):
        print("⚡ Cache Found: Loading Vector Index from Disk...")
        vectorstore = FAISS.load_local(
            index_folder, 
            embeddings, 
            allow_dangerous_deserialization=True # Safe since we created the index
        )
        
    # Scenario B: First time running. Do the work.
    else:
        print("🐢 No Cache: Processing PDF (This happens only once)...")
        # 1. Load
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # 3. Embed & Index
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # 4. Save for next time
        vectorstore.save_local(index_folder)
        print("💾 Index Saved to Disk for future runs.")
    
    # --- SEARCH LOGIC (Same as before) ---
    query = state['user_request']
    relevant_chunks = vectorstore.similarity_search(query, k=4)
    
    insights = []
    for i, chunk in enumerate(relevant_chunks):
        insights.append({
            "source_document": pdf_path,
            "page_number": chunk.metadata.get('page', 0),
            "key_finding": f"Relevant Extract {i+1}",
            "implication": chunk.page_content 
        })
        
    print(f"--- Retrieved {len(insights)} relevant passages ---")
    return {"pdf_insights": insights}

def writer_node(state: ResearchState):
    print("--- WRITER: Generating Final Report ---")
    
    # 1. Gather Context
    web_data = "\n".join([f"- {item['title']}: {item['snippet']}" for item in state['web_findings']])
    pdf_data = "\n".join([f"- {item['implication']}" for item in state['pdf_insights']])
    
    # 2. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 3. Construct Prompt
    system_msg = SystemMessage(content="""
        You are a Senior Data Analyst for a Fintech Enterprise. 
        Your goal is to synthesize data from web searches and internal PDF documents into a concise executive email.
        
        Rules:
        1. Start with a "Bottom Line Up Front" (BLUF).
        2. Use bullet points for key findings.
        3. Clearly distinguish between "Market Trends" (from Web) and "Regulatory Notes" (from PDF).
        4. Tone: Professional, direct, actionable. No fluff.
    """)
    
    user_msg = HumanMessage(content=f"""
        USER REQUEST: {state['user_request']}
        
        === WEB RESEARCH DATA ===
        {web_data}
        
        === INTERNAL PDF DATA ===
        {pdf_data}
        
        Write the email now.
    """)
    
    # 4. Invoke Model
    response = llm.invoke([system_msg, user_msg])
    
    return {"final_report": {"email_draft": response.content}}

# --- 2. DEFINE ROUTING LOGIC (The Brain) ---
# CRITICAL: This function must be defined BEFORE the graph uses it.

def router(state: ResearchState):
    # The Manager has already decided. We just execute.
    decision = state.get("current_step")
    
    if decision == "web_researcher":
        return "web_researcher"
    elif decision == "pdf_analyst":
        return "pdf_analyst"
    elif decision == "writer":
        return "writer"
    else:
        # Fallback if something breaks
        return "writer"

# --- 3. BUILD THE GRAPH (The Plumbing) ---

workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("manager", manager_node)
workflow.add_node("web_researcher", web_research_node)
workflow.add_node("pdf_analyst", pdf_analyst_node)
workflow.add_node("writer", writer_node)

# Set Entry Point
workflow.set_entry_point("manager")

# Add Conditional Edges
workflow.add_conditional_edges(
    "manager",
    router, 
    {
        "web_researcher": "web_researcher",
        "pdf_analyst": "pdf_analyst",
        "writer": "writer"
    }
)

# Add Normal Edges
workflow.add_edge("web_researcher", "manager")
workflow.add_edge("pdf_analyst", "manager")
workflow.add_edge("writer", END)

# Compile
app = workflow.compile()

# --- 4. EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- STARTING AGENT WORKFLOW ---")
    
    initial_state = {
        "user_request": "Research AI adoption in Indian fintech",
        "web_findings": [],
        "pdf_insights": [],
        "iterations": 0
    }
    
    result = app.invoke(initial_state)
    
    print("--- WORKFLOW FINISHED ---")
    
    final_email = result["final_report"]["email_draft"]
    print("\n" + "="*50)
    print("FINAL GENERATED EMAIL")
    print("="*50)
    print(final_email)
    print("="*50)