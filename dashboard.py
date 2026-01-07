import streamlit as st
import os
import shutil # Added for folder cleanup
from typing import List, Literal, Optional, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from tavily import TavilyClient

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Enterprise AI Researcher", page_icon="🕵️‍♂️", layout="wide")
load_dotenv()

# Check API Keys
if not os.getenv("TAVILY_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Missing API Keys. Please check your .env file.")
    st.stop()

# --- 2. STATE DEFINITION (The Memory) ---
class ResearchState(TypedDict):
    user_request: str
    web_findings: List[dict]
    pdf_insights: List[dict]
    current_step: str
    final_report: dict

# --- 3. AGENT NODES (The Workers) ---

def manager_node(state: ResearchState):
    # Define Decision Schema
    class ManagerDecision(BaseModel):
        next_step: Literal["web_researcher", "pdf_analyst", "writer"] = Field(description="Next agent.")
        reasoning: str = Field(description="Why?")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ManagerDecision)
    
    web_status = "Done" if state.get('web_findings') else "Not Started"
    pdf_status = "Done" if state.get('pdf_insights') else "Not Started"
    
    system_msg = SystemMessage(content="""
        You are the Research Orchestrator. Decide the next step.
        1. 'web_researcher': If we need external market data.
        2. 'pdf_analyst': If we need internal document data.
        3. 'writer': ONLY when we have sufficient data from BOTH sources (or one is skipped).
        Do not repeat steps.
    """)
    user_msg = HumanMessage(content=f"Request: {state['user_request']}\nStatus: Web={web_status}, PDF={pdf_status}")
    
    decision = structured_llm.invoke([system_msg, user_msg])
    return {"current_step": decision.next_step}

def web_research_node(state: ResearchState):
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(query=state['user_request'], search_depth="advanced", max_results=3)
    
    findings = [{"title": r['title'], "snippet": r['content'][:200], "url": r['url']} for r in response['results']]
    return {"web_findings": findings}

def pdf_analyst_node(state: ResearchState):
    pdf_path = "source_doc.pdf"
    index_folder = "faiss_index_ui" # Distinct folder for UI cache
    
    if not os.path.exists(pdf_path):
        return {"pdf_insights": []}
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Caching Logic
    if os.path.exists(index_folder):
        vectorstore = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(index_folder)
    
    relevant_chunks = vectorstore.similarity_search(state['user_request'], k=4)
    insights = [{"implication": c.page_content, "page": c.metadata.get('page')} for c in relevant_chunks]
    return {"pdf_insights": insights}

def writer_node(state: ResearchState):
    web_data = "\n".join([f"- {i['title']}: {i['snippet']}" for i in state['web_findings']])
    pdf_data = "\n".join([f"- {i['implication']}" for i in state['pdf_insights']])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msg = HumanMessage(content=f"""
        Write an Executive Email based on this:
        REQUEST: {state['user_request']}
        WEB DATA: {web_data}
        PDF DATA: {pdf_data}
    """)
    response = llm.invoke([msg])
    return {"final_report": {"email_draft": response.content}}

def router(state: ResearchState):
    return state["current_step"]

# --- 4. BUILD GRAPH ---
workflow = StateGraph(ResearchState)
workflow.add_node("manager", manager_node)
workflow.add_node("web_researcher", web_research_node)
workflow.add_node("pdf_analyst", pdf_analyst_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("manager")
workflow.add_conditional_edges("manager", router, {
    "web_researcher": "web_researcher",
    "pdf_analyst": "pdf_analyst",
    "writer": "writer"
})
workflow.add_edge("web_researcher", "manager")
workflow.add_edge("pdf_analyst", "manager")
workflow.add_edge("writer", END)
app = workflow.compile()

# --- 5. UI LAYOUT (The Dashboard) ---
st.set_page_config(page_title="AjayDataLabs | Enterprise Researcher", page_icon="🚀", layout="wide")

# Custom Branding Header
st.markdown("""
    <style>
    .main-title {font-size: 3rem; font-weight: 700; color: #4F8BF9;}
    .sub-title {font-size: 1.2rem; font-weight: 400; color: #555;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title"> AjayDataLabs Enterprise Researcher</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Autonomous Multi-Agent System: LangGraph Orchestrator + Tavily Web Search + Local RAG</p>', unsafe_allow_html=True)
st.divider()

# Sidebar for Inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8654/8654209.png", width=80)
    st.header("🗂️ Project Workspace")
    st.caption("Powered by **AjayDataLabs**")
    
    uploaded_file = st.file_uploader("Upload Company PDF", type=["pdf"])
    
    if uploaded_file:
        with open("source_doc.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Clear old cache if new file uploaded
        if os.path.exists("faiss_index_ui"):
            shutil.rmtree("faiss_index_ui")
        st.success("✅ Document Indexed")
        
    query = st.text_area("Research Topic", height=100, value="Analyze AI risks in this document and compare with current market trends.")
    
    st.divider()
    run_btn = st.button("🚀 Launch Agents", type="primary")
    
    st.markdown("---")
    st.markdown("# 🕸️") # or 🧠 or 🔗
    st.info("""
        **Ajay**
        *Senior Data Scientist & AI Engineer*
        
        [LinkedIn Profile](https://www.linkedin.com/in/ajaydatalabs-90b59932b) | [GitHub](https://github.com/Ajay2478)
    """)
    
# Main Execution Area
if run_btn:
    col1, col2 = st.columns([1, 1])
    
    final_state = None 
    
    with col1:
        st.subheader("⚙️ Live Agent Logs")
        log_expander = st.expander("View Full Trace", expanded=True)
        
        initial_state = {
            "user_request": query,
            "web_findings": [],
            "pdf_insights": [],
            "current_step": "manager",
            "iterations": 0
        }
        
        # Run!
        try:
            with st.spinner("Agents are collaborating..."):
                final_state = app.invoke(initial_state)
                
                with log_expander:
                    st.write("Manager Planned Tasks...")
                    if final_state['web_findings']:
                        st.info(f"🌐 Web Agent: Found {len(final_state['web_findings'])} sources")
                    if final_state['pdf_insights']:
                        st.warning(f"📄 PDF Agent: Extracted {len(final_state['pdf_insights'])} key segments")
                    st.success("✅ Writer Agent: Report Generated")
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
    
    with col2:
        st.subheader("📝 Final Executive Report")
        if final_state and "final_report" in final_state:
            report = final_state["final_report"]["email_draft"]
            st.markdown(report)
            st.download_button("Download Report", report, file_name="AjayDataLabs_Report.txt")
        else:
            st.info("Report will appear here after agents finish.")