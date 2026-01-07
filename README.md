# 🕵️‍♂️ AjayDataLabs | Multi-Agent Enterprise Researcher
**Autonomous AI System for Deep Market Research & Document Analysis**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![RAG](https://img.shields.io/badge/RAG-Local%20Embeddings-green)](https://huggingface.co/)

> *"An intelligent agentic workflow that autonomously plans research strategies, scrapes the live web, reads internal PDFs, and synthesizes executive reports."*

---

## 🚀 Overview

Standard RAG (Retrieval-Augmented Generation) systems are brittle—they fail if the document lacks the answer. **AjayDataLabs Enterprise Researcher** solves this by implementing a **Supervisor-Worker Architecture**.

It does not just "chat" with a PDF. It reasons.
1.  **Manager Agent:** Analyzes the user request and decides: *"Do I need external market data, internal document data, or both?"*
2.  **Web Agent:** Uses **Tavily API** to fetch high-credibility real-time news.
3.  **Internal Agent:** Uses **Local Embeddings (HuggingFace)** to vector-search private documents.
4.  **Writer Agent:** Synthesizes conflicting data points into a coherent "Bottom Line Up Front" (BLUF) email.

## ⚡ Key Technical Features

### 🏗️ Autonomous Orchestration (LangGraph)
Unlike linear chains, this system uses a **State Machine**. The Manager agent dynamically routes tasks based on data gaps. If the web search is insufficient, it can trigger a document review, and vice versa.

### 💰 Zero-Cost Local RAG
To optimize cloud costs, this project implements a hybrid approach:
* **LLM (Logic):** GPT-4o-mini (Low cost, high reasoning).
* **Embeddings (Memory):** `all-MiniLM-L6-v2` (Running locally via HuggingFace).
* **Vector Database:** FAISS (Facebook AI Similarity Search).

This architecture allows for **free** processing of heavy documents without hitting OpenAI API rate limits.

### 🏎️ Performance Engineering
* **Vector Caching:** The system computes embeddings once and serializes them to disk. Subsequent queries on the same document are **95% faster** (Latency drop: 20s → <1s).
* **Structured Output:** All agents communicate via strict Pydantic schemas, preventing JSON parsing errors in production.

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph
* **Frontend:** Streamlit
* **Web Search:** Tavily AI
* **Vector Store:** FAISS
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **LLM:** OpenAI GPT-4o-mini

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Ajay2478/Enterprise-AI-Researcher.git
cd Enterprise-AI-Researcher
