# AgentRAG — RAG Agent with Tool Calling

Intelligent ReAct agent that reasons step by step and calls the right tool to answer any question. Combines RAG retrieval with web search, calculator, datetime, and more.

## How ReAct Works
```
User Question
      ↓
Thought: What do I need to find out?
Action: search_knowledge_base("pricing plans")
Observation: TaskFlow Pro costs $29/user/month...
      ↓
Thought: Now I need to calculate annual cost.
Action: calculate(29 * 12 * 10)
Observation: 29 * 12 * 10 = 3480
      ↓
Final Answer: For 10 users on Pro plan, annual cost is $3,480.
```

## Tech Stack

- Python
- Groq API (Free)
- LangChain + LangChain Text Splitters
- FAISS (local vector store)
- HuggingFace all-MiniLM-L6-v2 (embeddings)
- DuckDuckGo Search (free web search)
- Streamlit

## 6 Tools

| Tool | Purpose |
|---|---|
| search_knowledge_base | RAG retrieval from uploaded documents |
| search_web | DuckDuckGo internet search |
| get_current_datetime | Current date and time |
| calculate | Math: +, -, *, /, **, sqrt, % |
| summarize | Condense retrieved content |
| check_knowledge_base | Check if topic exists in KB |

## Files

- `app.py` — Streamlit chat interface with ReAct trace viewer
- `AgentRAG.ipynb` — Step-by-step notebook (12 sections)
- `requirements.txt` — All dependencies

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Get Free API Key
https://console.groq.com
