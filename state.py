from typing import List, Optional, Annotated
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output (The Contract) ---

class WebSource(BaseModel):
    url: str
    title: str
    snippet: str
    credibility_score: int = Field(description="1-10 score of source reliability")

class PDFInsight(BaseModel):
    source_document: str
    page_number: int
    key_finding: str
    implication: str

class ExecutiveReport(BaseModel):
    executive_summary: str
    strategic_recommendations: List[str]
    risk_analysis: str
    email_draft: str

# --- The Graph State (The Memory) ---

class ResearchState(TypedDict):
    user_request: str
    web_findings: List[dict]  # Simplified for now
    pdf_insights: List[dict]
    current_step: str         # <--- WE ADDED THIS
    final_report: dict