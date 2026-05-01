---
title: Multi Agent Book Integrity RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Integrity-Driven Book Recommender: Multi-Agent RAG with Hallucination Detection & Self-Correction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(Llama%203.2)-orange.svg)](https://ollama.com/)

An elevated **Retrieval-Augmented Generation (RAG)** pipeline designed to mitigate hallucinations using a multi-agent collaborative framework. This project simulates **Metacognitive** processes by implementing a "Checks and Balances" architecture to ensure AI responses are trustworthy and strictly based on the provided data.

## Key Features

- **Multi-Agent Collaboration**: Features three specialized agents (The Librarian, The Grader, and The Editor).
- **Self-Correction Logic**: Automatically detects and fixes hallucinations before the user sees the output.
- **Rigid Integrity Guardrails**: Prevents logical leaps and ensures the AI accepts ignorance when data is missing.
- **Local & Private**: Powered by **Llama 3.2** running locally via Ollama.

## Architecture: The Agentic Loop

This system transitions from a linear RAG to a **dynamic loop** that mirrors human deliberative thinking:

1.  **The Librarian (Generator)**: Performs semantic search on the dataset and generates an initial draft.
2.  **The Grader (Auditor)**: Critically evaluates the draft against the raw CSV context. It checks for intent matching and factual grounding.
3.  **The Editor (Refiner)**: Receives feedback from the Grader to prune non-factual info and rewrites the response.

---

## Evaluation & Benchmark

The system is aim to handle common RAG failure modes, such as Out-of-context queries where standard LLMs often hallucinate.

| Scenario | User Query | Naive RAG Result | Agentic RAG (This System) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Negative Constraint** | "Book about a green elephant?" | Might hallucinate a description. | **Caught by Grader.** Editor admits no data found. |  Passed |
| **Topic Shift** | "Is there a Rainbow Crow?" | Discusses generic birds. | **Rejected.** Identifies specific topic as missing. |  Passed |
| **Factual Retrieval** | "Tell me about 'The Hours'" | Accurate description. | **Verified.** Confirmed against CSV metadata. |  Passed |

Current Limitations: Mention that the 3B model occasionally relies on pre-trained knowledge for common titles (e.g., Children's books) and I am working on stricter attribute filtering (e.g., gender).

---

## Installation & Setup

### Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- Python 3.11 or higher.

### Setup
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/agentic-rag-integrity.git](https://github.com/yourusername/agentic-rag-integrity.git)
   cd agentic-rag-integrity
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
3. **Pull the model**
    ```bash
    ollama pull llama3.2
4. Run main.py
    ```bash
    python main.py
