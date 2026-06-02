import streamlit as st
import os
import shutil
from git import Repo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Source Code Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main Theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Custom Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
        border-color: #58a6ff;
    }
    
    /* Chat Bubbles */
    .chat-bubble {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #1f6feb;
        margin-left: auto;
    }
    .assistant-bubble {
        background-color: #21262d;
        border: 1px solid #30363d;
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        padding: 10px;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ea043, #238636);
        box-shadow: 0px 0px 15px rgba(46, 160, 67, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "repo_summary" not in st.session_state:
    st.session_state.repo_summary = None
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None

# --- Logic from app.ipynb ---

def process_repository(repo_url):
    # Extract repo name
    repo_name = repo_url.split("/")[-1]
    repo_name = repo_name.replace(".git", "")
    
    # Local storage path
    repo_path = f"./repos/{repo_name}"
    
    # Create repos directory
    os.makedirs("repos", exist_ok=True)
    
    # Clone Repository
    if not os.path.exists(repo_path):
        with st.status("🚀 Cloning repository...", expanded=True) as status:
            Repo.clone_from(repo_url, repo_path)
            status.update(label="✅ Repository cloned successfully", state="complete")
    else:
        st.info("📦 Repository already exists locally")

    # Load Files
    documents = []
    py_count = 0
    md_count = 0
    txt_count = 0
    supported_extensions = (".py", ".md", ".txt")

    with st.status("📂 Loading and processing files...", expanded=False) as status:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    if file.endswith(".py"): py_count += 1
                    elif file.endswith(".md"): md_count += 1
                    elif file.endswith(".txt"): txt_count += 1
                    
                    try:
                        loader = TextLoader(file_path, encoding="utf-8")
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["file_name"] = file
                            doc.metadata["repo_name"] = repo_name
                        documents.extend(docs)
                    except Exception as e:
                        st.warning(f"⚠️ Skipped: {file_path} - {e}")
        status.update(label=f"✅ Loaded {len(documents)} documents", state="complete")

    # Chunking
    splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitters.split_documents(documents)
    
    # Embedding
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Vector Store
    # Use repo specific chroma_db to avoid mixing
    persist_dir = f"./chroma_db/{repo_name}"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir) # Clear old data for fresh start if re-processing
        
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir,
    )
    
    # Retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )
    
    return {
        "retriever": retriever,
        "stats": {
            "name": repo_name,
            "py": py_count,
            "md": md_count,
            "txt": txt_count,
            "count": len(documents)
        }
    }

def ask_question(retriever, question):
    model_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    prompt_template = PromptTemplate.from_template(
        "\"\"\" You are a senior software engineer and repository analyst.\n\n"
        "Your task is to analyze a GitHub repository.\n\n"
        "Instructions:\n"
        "- Answer only from the provided repository context.\n"
        "- If the user asks about tech stack, analyze dependencies, imports, and README.\n"
        "- If the user asks about architecture, infer from the files.\n"
        "- Mention file names whenever possible.\n"
        "- Use bullet points.\n"
        "- If information is missing, explicitly say so.\n\n"
        "Repository Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}\"\"\""
    )
    
    # Retrieve docs
    retrieved_docs = retriever.invoke(question)
    
    # Context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Build prompt
    prompt = prompt_template.format(context=context, question=question)
    
    # LLM call
    response = model_llm.invoke(prompt)
    
    # Sources
    sources = list(set(doc.metadata["file_name"] for doc in retrieved_docs))
    
    return {"answer": response.content, "sources": sources}

# --- UI Components ---

st.sidebar.markdown("# 🧠 GitHub Code Analyzer ")
st.sidebar.markdown("Analyze any GitHub repository with AI power.")

repo_url = st.sidebar.text_input("GitHub URL", placeholder="https://github.com/user/repo")
process_btn = st.sidebar.button("Process Repository")

if process_btn and repo_url:
    result = process_repository(repo_url)
    st.session_state.retriever = result["retriever"]
    st.session_state.stats = result["stats"]
    st.session_state.repo_name = result["stats"]["name"]
    # Clear previous chat
    st.session_state.messages = []
    st.sidebar.success(f"Processed: {st.session_state.repo_name}")

# Main UI
if st.session_state.retriever:
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card">📂 Files<br><h3>{st.session_state.stats["count"]}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card">🐍 Python<br><h3>{st.session_state.stats["py"]}</h3></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card">📝 Markdown<br><h3>{st.session_state.stats["md"]}</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card">📄 Text<br><h3>{st.session_state.stats["txt"]}</h3></div>', unsafe_allow_html=True)

    st.divider()

    # Tabs for Chat and Summary
    tab_chat, tab_summary = st.tabs([" Code Chat", " Repo Summary"])

    with tab_chat:
        st.markdown(f"### Chat with **{st.session_state.repo_name}**")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    st.caption(f"Sources: {', '.join(message['sources'])}")

        # Chat Input
        if prompt := st.chat_input("Ask about the codebase..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_question(st.session_state.retriever, prompt)
                    st.markdown(response["answer"])
                    st.caption(f"Sources: {', '.join(response['sources'])}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response["sources"]
                    })

    with tab_summary:
        if st.button("Generate Complete Summary"):
            with st.spinner("Analyzing repository..."):
                summary_q = """
                Analyze this repository and provide:
                1. Project Purpose (One clear paragraph)
                2. Core Tech Stack (Bullet points)
                3. Architecture Overview
                4. Key Features & Functionalities
                5. Important Files & their roles
                """
                result = ask_question(st.session_state.retriever, summary_q)
                st.markdown(result["answer"])
                st.info(f"Summary based on: {', '.join(result['sources'])}")

else:
    # Welcome Screen
    st.markdown("<h1 style='text-align: center;'>AI Source Code Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; opacity: 0.8;'>Unlock the secrets of any GitHub repository in seconds.</p>", unsafe_allow_html=True)
    
    # st.image("https://images.unsplash.com/photo-1555066931-4365d14bab8c?auto=format&fit=crop&q=80&w=1000", use_container_width=True)
    
    st.markdown("""
    ### How it works:
    1. **📥 Enter URL**: Paste the GitHub repository link in the sidebar.
    2. **⚙️ Process**: We clone, chunk, and embed the code into a vector database.
    3. **💬 Chat**: Ask questions about architecture, bugs, or logic.
    4. **📊 Summarize**: Get a comprehensive overview of the project structure and tech stack.
    """)

    st.warning("👈 Please enter a GitHub URL in the sidebar to get started.")

# Footer
st.markdown("---")

