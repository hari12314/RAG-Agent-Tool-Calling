import streamlit as st
import os
import re
import math
import time
import json
import tempfile
from datetime import datetime
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="AgentRAG", page_icon="A", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@300;400;500&display=swap');

:root {
    --bg:     #07080f;
    --s1:     #0e0f1a;
    --s2:     #141525;
    --s3:     #1c1d30;
    --border: #252640;
    --acc:    #00d4ff;
    --acc2:   #ff6b35;
    --green:  #00ff87;
    --purple: #9d4edd;
    --yellow: #ffd60a;
    --text:   #e8eaf6;
    --muted:  #5c5f80;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.hero {
    background: linear-gradient(135deg, #0e0f1a 0%, #141525 50%, #1a1535 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--acc), var(--purple), var(--acc2));
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
    margin: 0;
}

.hero-title .a { color: var(--acc); }
.hero-title .r { color: var(--acc2); }
.hero-sub { font-family: 'Fira Code', monospace; font-size: 0.7rem; color: var(--muted); margin-top: 0.4rem; letter-spacing: 0.05em; }

.chat-area {
    height: 560px;
    overflow-y: auto;
    padding: 1rem;
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.msg-user { display: flex; justify-content: flex-end; }

.msg-agent { display: flex; gap: 0.7rem; align-items: flex-start; }

.bubble-user {
    background: linear-gradient(135deg, var(--acc), #0099cc);
    color: #07080f;
    font-weight: 500;
    border-radius: 16px 16px 4px 16px;
    padding: 0.7rem 1.1rem;
    max-width: 72%;
    font-size: 0.9rem;
    line-height: 1.6;
}

.bubble-agent {
    background: var(--s2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.1rem;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.7;
}

.agent-avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--acc), var(--purple));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.85rem; color: white;
    flex-shrink: 0; margin-top: 2px;
}

.step-trace {
    background: var(--s3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--purple);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    margin-top: 0.6rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    line-height: 1.7;
}

.step-trace .thought { color: var(--yellow); }
.step-trace .action  { color: var(--acc); }
.step-trace .obs     { color: var(--green); }

.tool-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--s3);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem;
    color: var(--acc);
    margin: 2px 3px 0 0;
}

.meta-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }

.meta-chip {
    background: var(--s3);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 9px;
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem;
    color: var(--muted);
}

.meta-chip span { color: var(--text); font-weight: 600; }

.tool-card {
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}

.tool-card .tname { font-family: 'Fira Code', monospace; color: var(--acc); font-size: 0.78rem; font-weight: 500; }
.tool-card .tdesc { color: var(--muted); font-size: 0.75rem; margin-top: 0.2rem; }

.kb-ready {
    background: linear-gradient(135deg, #0a1f0a, #0d2a0d);
    border: 1px solid #1a4a1a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.75rem;
    color: var(--green);
    margin-top: 0.5rem;
}

.stTextInput input {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem !important;
}

.stTextInput input:focus { border-color: var(--acc) !important; }

.stSelectbox > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #9d4edd22) !important;
    color: var(--acc) !important;
    border: 1px solid var(--acc) !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #9d4edd44) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# Session init
for k, v in [("messages",[]),("vs",None),("kb_built",False),("chunks",0),("doc_names",[])]:
    if k not in st.session_state: st.session_state[k] = v

TOOL_INFO = [
    ("search_knowledge_base", "RAG search from uploaded documents"),
    ("search_web",            "DuckDuckGo internet search (free)"),
    ("get_current_datetime",  "Current date and time"),
    ("calculate",             "Math: +, -, *, /, **, sqrt, %"),
    ("summarize",             "Summarize retrieved content"),
    ("check_knowledge_base",  "Check if topic exists in KB"),
]

TOOL_COLORS = {"search_knowledge_base":"#00d4ff","search_web":"#ff6b35",
               "get_current_datetime":"#ffd60a","calculate":"#00ff87",
               "summarize":"#9d4edd","check_knowledge_base":"#ff87ab"}

@st.cache_resource(show_spinner=False)
def get_emb():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

def load_file(f):
    suffix = ".pdf" if f.type=="application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.read()); path=tmp.name
    loader = PyPDFLoader(path) if suffix==".pdf" else TextLoader(path, encoding="utf-8")
    pages = loader.load()
    for p in pages: p.metadata["source"] = f.name
    os.unlink(path)
    return pages

def build_index(docs, cs, co, emb):
    sp = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=co, separators=["\n\n","\n",". "," ",""])
    chunks = sp.split_documents(docs)
    return FAISS.from_documents(chunks, emb), len(chunks)

# Tool functions
def t_kb(vs, q, k=3):
    if vs is None: return "No knowledge base loaded."
    r = vs.similarity_search(q, k=k)
    return "\n\n".join([x.page_content for x in r]) if r else "No results found."

def t_web(q, n=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            results = list(d.text(q, max_results=n))
        return "\n\n".join([r.get("title","") + ": " + r.get("body","")[:200] for r in results]) if results else "No results."
    except Exception as e:
        return "Web search error: " + str(e)

def t_dt():
    n=datetime.now()
    return "Date: " + n.strftime("%A, %B %d, %Y") + "\nTime: " + n.strftime("%I:%M %p")

def t_calc(expr):
    try:
        allowed = {"sqrt":math.sqrt,"round":round,"abs":abs,"pow":pow,"pi":math.pi,"e":math.e}
        result = eval(expr.replace("%","/100"),{"__builtins__":{}},allowed)
        return str(expr) + " = " + str(round(float(result),4))
    except Exception as e:
        return "Calc error: " + str(e)

def t_sum(client, model, text):
    r = client.chat.completions.create(model=model,
        messages=[{"role":"user","content":"Summarize in 2 sentences:\n\n"+text[:1500]+"\n\nSUMMARY:"}],
        temperature=0.1, max_tokens=150)
    return r.choices[0].message.content.strip()

def t_check(vs, topic):
    if vs is None: return "No KB loaded."
    r = vs.similarity_search(topic, k=1)
    return ("YES - Found: " + r[0].page_content[:120]) if r else "NO - Not found."

def execute(client, model, vs, tool, args):
    t = tool.lower().strip()
    if t=="search_knowledge_base":   return t_kb(vs, args)
    elif t=="search_web":            return t_web(args)
    elif t=="get_current_datetime":  return t_dt()
    elif t=="calculate":             return t_calc(args)
    elif t=="summarize":             return t_sum(client, model, args)
    elif t=="check_knowledge_base":  return t_check(vs, args)
    else: return "Unknown tool: " + tool

def parse_action(text):
    m = re.search(r'Action:\s*(\w+)\((.*)\)', text, re.DOTALL)
    if m: return m.group(1).strip(), m.group(2).strip().strip('"').strip("'")
    return None, None

def build_system_prompt():
    return (
        "You are AgentRAG, an intelligent assistant with tools.\n"
        "Use ReAct format: Thought -> Action -> Observation -> repeat -> Final Answer.\n\n"
        "FORMAT:\nThought: [reasoning]\nAction: tool_name(argument)\n"
        "Observation: [inserted by system]\n...\nFinal Answer: [answer]\n\n"
        "TOOLS:\n"
        "1. search_knowledge_base(query) - Search uploaded documents\n"
        "2. search_web(query) - Internet search\n"
        "3. get_current_datetime() - Current date/time\n"
        "4. calculate(expression) - Math calculations\n"
        "5. summarize(text) - Summarize content\n"
        "6. check_knowledge_base(topic) - Check if topic is in KB\n\n"
        "RULES: Always start with Thought. One tool per Action. End with Final Answer."
    )

def run_agent(client, model, vs, question, max_steps=6, temperature=0.1):
    t0 = time.time()
    steps, tools_used = [], []
    sys_prompt = build_system_prompt()
    messages = [
        {"role":"system","content":sys_prompt},
        {"role":"user","content":"Question: " + question},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=500,
            stop=["Observation:"],
        )
        out = response.choices[0].message.content.strip()

        if "Final Answer:" in out:
            final = out.split("Final Answer:")[-1].strip()
            steps.append({"step":step+1,"type":"final","content":out})
            return {"answer":final,"steps":steps,"tools_used":list(set(tools_used)),"total_steps":step+1,"time":round(time.time()-t0,2)}

        tool, args = parse_action(out)
        if tool:
            tools_used.append(tool)
            obs = execute(client, model, vs, tool, args)
            steps.append({"step":step+1,"type":"action","thought":out,"tool":tool,"args":args,"observation":obs})
            messages.append({"role":"assistant","content":out})
            messages.append({"role":"user","content":"Observation: " + obs + "\n\nContinue:"})
        else:
            messages.append({"role":"assistant","content":out})
            messages.append({"role":"user","content":"Please provide your Final Answer now."})

    return {"answer":"Reached max steps.","steps":steps,"tools_used":list(set(tools_used)),"total_steps":max_steps,"time":round(time.time()-t0,2)}

# SIDEBAR
with st.sidebar:
    st.markdown('<div style="font-family:Outfit,sans-serif;font-size:1.3rem;font-weight:800;color:#e8eaf6;letter-spacing:-0.5px;"><span style="color:#00d4ff;">Agent</span><span style="color:#ff6b35;">RAG</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Fira Code,monospace;font-size:0.68rem;color:#5c5f80;margin-bottom:1rem;">ReAct Agent · Tool Calling</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Key**")
    st.caption("Free at: https://console.groq.com")
    api_key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")

    st.markdown("**Model**")
    model = st.selectbox("", ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768"], label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_steps   = st.slider("Max Steps", 3, 10, 6)

    st.markdown("**Chunking**")
    chunk_size    = st.slider("Chunk Size", 200, 800, 300, 50)
    chunk_overlap = st.slider("Overlap",      0, 150,  40, 10)

    st.markdown("---")
    st.markdown("**Available Tools**")
    for tname, tdesc in TOOL_INFO:
        color = TOOL_COLORS.get(tname, "#ffffff")
        st.markdown(
            '<div class="tool-card">'
            '<div class="tname" style="color:' + color + ';">' + tname + '</div>'
            '<div class="tdesc">' + tdesc + '</div>'
            '</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state["messages"] = []
        st.rerun()

# HEADER
st.markdown(
    '<div class="hero"><h1 class="hero-title"><span class="a">Agent</span><span class="r">RAG</span></h1>'
    '<p class="hero-sub">ReAct PROMPTING &nbsp;|&nbsp; MULTI-TOOL REASONING &nbsp;|&nbsp; KNOWLEDGE BASE + WEB + CALCULATOR</p></div>',
    unsafe_allow_html=True)

left, right = st.columns([1, 1.5], gap="large")

# LEFT: Upload
with left:
    st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.68rem;color:#00d4ff;text-transform:uppercase;letter-spacing:0.12em;">Knowledge Base</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["pdf","txt"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded:
        if st.button("Build Knowledge Base"):
            if not api_key:
                st.error("Enter your Groq API key.")
            else:
                with st.spinner("Loading embeddings..."):
                    emb = get_emb()
                all_pages = []
                with st.spinner("Reading files..."):
                    for f in uploaded: all_pages.extend(load_file(f))
                with st.spinner("Indexing..."):
                    vs, nc = build_index(all_pages, chunk_size, chunk_overlap, emb)
                st.session_state["vs"]        = vs
                st.session_state["kb_built"]  = True
                st.session_state["chunks"]    = nc
                st.session_state["doc_names"] = [f.name for f in uploaded]
                st.session_state["messages"]  = []
                st.success("Knowledge base ready!")

    if st.session_state["kb_built"]:
        st.markdown(
            '<div class="kb-ready">'
            + str(len(st.session_state["doc_names"])) + ' doc(s) &nbsp;|&nbsp; '
            + str(st.session_state["chunks"]) + ' chunks &nbsp;|&nbsp; FAISS indexed'
            + '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.68rem;color:#00d4ff;text-transform:uppercase;letter-spacing:0.12em;">ReAct Flow</p>', unsafe_allow_html=True)

    flow = [("T","Thought","Agent reasons about what to do","#ffd60a"),
            ("A","Action","Calls the right tool","#00d4ff"),
            ("O","Observation","Gets tool result","#00ff87"),
            ("R","Repeat","Until answer found","#9d4edd"),
            ("F","Final Answer","Responds to user","#ff6b35")]

    for sym, title, desc, color in flow:
        st.markdown(
            '<div style="background:var(--s2);border:1px solid var(--border);border-radius:8px;'
            'padding:0.5rem 0.9rem;margin-bottom:0.3rem;display:flex;gap:0.8rem;align-items:center;">'
            '<div style="width:26px;height:26px;border-radius:6px;background:' + color + '22;'
            'border:1px solid ' + color + '44;display:flex;align-items:center;justify-content:center;'
            'font-weight:800;font-size:0.75rem;color:' + color + ';flex-shrink:0;">' + sym + '</div>'
            '<div><strong style="font-size:0.82rem;color:#e8eaf6;">' + title + '</strong>'
            '<span style="font-size:0.74rem;color:#5c5f80;margin-left:0.5rem;">' + desc + '</span>'
            '</div></div>', unsafe_allow_html=True)

# RIGHT: Chat
with right:
    st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.68rem;color:#00d4ff;text-transform:uppercase;letter-spacing:0.12em;">Agent Chat</p>', unsafe_allow_html=True)

    # Build chat HTML
    chat_html = '<div class="chat-area">'

    if not st.session_state["messages"]:
        chat_html += (
            '<div class="msg-agent"><div class="agent-avatar">AR</div>'
            '<div class="bubble-agent">'
            'Hello! I am <strong>AgentRAG</strong> — a ReAct agent with multiple tools.<br><br>'
            'I can:<br>'
            '&nbsp;&nbsp;- Search your knowledge base (RAG)<br>'
            '&nbsp;&nbsp;- Search the web (DuckDuckGo)<br>'
            '&nbsp;&nbsp;- Do math calculations<br>'
            '&nbsp;&nbsp;- Get current date/time<br>'
            '&nbsp;&nbsp;- Summarize retrieved content<br><br>'
            + ("Upload documents on the left to get started, or ask me anything!" if not st.session_state["kb_built"]
               else "Knowledge base loaded! Ask me anything.") +
            '</div></div>'
        )
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                chat_html += '<div class="msg-user"><div class="bubble-user">' + msg["content"] + '</div></div>'
            else:
                tool_badges = "".join([
                    '<span class="tool-badge" style="color:' + TOOL_COLORS.get(t,"#fff") + ';border-color:' + TOOL_COLORS.get(t,"#444") + '44;">' + t + '</span>'
                    for t in msg.get("tools_used",[])
                ])
                chat_html += (
                    '<div class="msg-agent"><div class="agent-avatar">AR</div>'
                    '<div style="max-width:85%;">'
                    '<div class="bubble-agent">' + msg["content"].replace("\n","<br>") + '</div>'
                )

                # Show step trace
                if msg.get("show_trace") and msg.get("steps"):
                    trace_html = '<div class="step-trace">'
                    for s in msg["steps"]:
                        if s["type"] == "action":
                            lines = s.get("thought","").split("\n")
                            for l in lines:
                                if l.startswith("Thought:"): trace_html += '<div class="thought">' + l + '</div>'
                                elif l.startswith("Action:"): trace_html += '<div class="action">' + l + '</div>'
                            trace_html += '<div class="obs">Observation: ' + s.get("observation","")[:150] + '...</div>'
                    trace_html += '</div>'
                    chat_html += trace_html

                chat_html += (
                    '<div class="meta-row">'
                    '<div class="meta-chip">Steps <span>' + str(msg.get("total_steps","")) + '</span></div>'
                    '<div class="meta-chip">Time <span>' + str(msg.get("time","")) + 's</span></div>'
                    + tool_badges +
                    '</div></div></div>'
                )

    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input
    c1, c2, c3 = st.columns([5, 1, 1])
    with c1:
        user_input = st.text_input("", placeholder="Ask anything — I will reason and use tools...", label_visibility="collapsed", key="agent_input")
    with c2:
        send_btn = st.button("Send")
    with c3:
        show_trace = st.toggle("Trace", value=True)

    if (send_btn or user_input) and user_input.strip():
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            groq_client = Groq(api_key=api_key)
            vs = st.session_state.get("vs")

            with st.spinner("Agent is reasoning..."):
                try:
                    result = run_agent(groq_client, model, vs, user_input.strip(), max_steps, temperature)
                    st.session_state["messages"].append({"role":"user","content":user_input.strip()})
                    st.session_state["messages"].append({
                        "role":"assistant",
                        "content":result["answer"],
                        "tools_used":result["tools_used"],
                        "total_steps":result["total_steps"],
                        "time":result["time"],
                        "steps":result["steps"],
                        "show_trace":show_trace,
                    })
                    st.rerun()
                except Exception as e:
                    st.error("Error: " + str(e))

    # Example prompts
    if not st.session_state["messages"]:
        examples = [
            "What is today's date?",
            "Calculate 29 * 12 for annual Pro cost",
            "Search the web for LangChain RAG",
        ]
        if st.session_state["kb_built"]:
            examples = [
                "What products are available and their pricing?",
                "Calculate annual cost for 10 users on Pro plan",
                "What integrations are supported?",
            ]
        st.markdown('<p style="font-family:Fira Code,monospace;font-size:0.63rem;color:#5c5f80;margin-top:0.6rem;">Try these:</p>', unsafe_allow_html=True)
        ecols = st.columns(len(examples))
        for i, (ec, ex) in enumerate(zip(ecols, examples)):
            with ec:
                if st.button(ex[:30] + "...", key="ex_"+str(i)):
                    if api_key:
                        groq_client = Groq(api_key=api_key)
                        with st.spinner("Thinking..."):
                            try:
                                r = run_agent(groq_client, model, st.session_state.get("vs"), ex, max_steps, temperature)
                                st.session_state["messages"].append({"role":"user","content":ex})
                                st.session_state["messages"].append({
                                    "role":"assistant","content":r["answer"],
                                    "tools_used":r["tools_used"],"total_steps":r["total_steps"],
                                    "time":r["time"],"steps":r["steps"],"show_trace":True,
                                })
                                st.rerun()
                            except Exception as e:
                                st.error("Error: " + str(e))
