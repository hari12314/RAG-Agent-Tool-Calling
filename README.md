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
| search_web | Internet search with DuckDuckGo (default) or optional Parallel Search MCP |
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

### Optional Parallel web search

The Streamlit app can use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for free web search in Fast mode, with source excerpts and URLs available to the
agent. No Parallel account or API key is required for light use. DuckDuckGo
remains the default; this option does not change the notebook.

After installing the app dependencies above, install the optional backend with
Python 3.11 or newer:

```bash
pip install -r requirements-parallel.txt
streamlit run app.py
```

Choose **Parallel** under **Web search provider** in the sidebar, enter your
Groq API key as usual, and try **Search the web for LangChain RAG**. The existing
`search_web` action sends the query to `https://search.parallel.ai/mcp` using
Streamable HTTP and passes the returned text to the agent as an observation.
Queries are sent to the selected provider. Knowledge-base retrieval continues
to use the existing FAISS index.

Search calls have a 30-second timeout. Free access has rate limits; failures
appear in the tool observation without switching providers. Select DuckDuckGo
to switch back. The Groq key is still needed for the agent's responses.

Backend and agent-dispatch checks can run without downloading embedding models:

```bash
python -m unittest discover -s tests -v
```

## Get Free API Key
https://console.groq.com
