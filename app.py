import streamlit as st
import os
import sys
# --- PATH FIX: Force Python to look in current directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# -----------------------------------------------------------
import shutil
from main import app  # Importing your LangGraph app

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise AI Researcher", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Multi-Agent Enterprise Researcher")
st.markdown("""
    **Architecture:** `LangGraph` Orchestrator | `Tavily` Web Search | `Local RAG` PDF Analysis
    \n*Upload a document, ask a question, and watch the agents collaborate.*
""")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload PDF (Optional)", type=["pdf"])
    
    # Logic to handle the file
    if uploaded_file:
        # Save the uploaded file as 'source_doc.pdf' so the agent can find it
        with open("source_doc.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("PDF Loaded to Knowledge Base!")
    
    st.divider()
    
    st.header("2. Define Task")
    # Default query to help the user start
    user_query = st.text_area(
        "Research Goal:", 
        value="Research AI adoption in Indian fintech and compare with the attached PDF.",
        height=100
    )
    
    run_btn = st.button("🚀 Start Research Agent", type="primary")

# --- MAIN EXECUTION ---
if run_btn:
    if not user_query:
        st.warning("Please enter a research topic.")
    else:
        # Create a container for the logs
        log_container = st.expander("🧐 Agent Thought Process (Logs)", expanded=True)
        result_container = st.container()
        
        # Initialize State
        initial_state = {
            "user_request": user_query,
            "web_findings": [],
            "pdf_insights": [],
            "current_step": "manager",
            "iterations": 0
        }

        # Run the Workflow
        with st.spinner("Agents are working... (Manager → Researcher → Analyst → Writer)"):
            try:
                # We invoke the graph
                # Note: For a true production app, we would use .stream() to show live updates.
                # For now, .invoke() is safer and simpler.
                final_state = app.invoke(initial_state)
                
                # Extract the Final Email
                final_report = final_state.get("final_report", {}).get("email_draft", "No report generated.")
                
                # --- DISPLAY LOGS (Reconstructing what happened) ---
                with log_container:
                    st.write("### 🧠 Execution Trace")
                    
                    # Show Web Findings if any
                    web_data = final_state.get("web_findings", [])
                    if web_data:
                        st.success(f"**Web Agent:** Found {len(web_data)} articles.")
                        for item in web_data:
                            st.caption(f"- {item['title']}")
                    
                    # Show PDF Findings if any
                    pdf_data = final_state.get("pdf_insights", [])
                    if pdf_data:
                        st.info(f"**PDF Agent:** Retrieved {len(pdf_data)} vector chunks.")
                        
                    st.write("---")
                    st.caption("Workflow completed successfully.")

                # --- DISPLAY FINAL RESULT ---
                with result_container:
                    st.subheader("📧 Final Executive Report")
                    st.markdown("---")
                    st.markdown(final_report)
                    st.markdown("---")
                    
                    # Add a download button for the email
                    st.download_button(
                        label="📥 Download Report",
                        data=final_report,
                        file_name="research_report.txt",
                        mime="text/plain"
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {e}")