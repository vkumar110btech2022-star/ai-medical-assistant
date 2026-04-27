import os
import html
import re
import logging
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
DATA_PATH = os.path.join(BASE_DIR, "data")


RAG_PROMPT_TEMPLATE = """You are a helpful medical reference assistant.
Use only the provided context to answer the question.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question:
{input}
"""

logger = logging.getLogger(__name__)

DEPRECATED_GROQ_MODELS = {
    "llama3-8b-8192": "llama-3.1-8b-instant",
    "llama3-70b-8192": "llama-3.3-70b-versatile",
}


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    ensure_vectorstore_exists(embedding_model)
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def ensure_vectorstore_exists(embedding_model):
    index_path = os.path.join(DB_FAISS_PATH, "index.faiss")
    metadata_path = os.path.join(DB_FAISS_PATH, "index.pkl")
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        return

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise RuntimeError("No PDF files found in data folder to build vectorstore.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)


def set_custom_prompt(custom_prompt_template):
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    return prompt


def is_general_chat_query(user_text):
    normalized = " ".join(user_text.strip().lower().split())
    if not normalized:
        return True

    words = normalized.split()
    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "yo",
        "hola",
        "good morning",
        "good afternoon",
        "good evening",
        "who are you",
    }

    # Treat greetings and very short queries like normal chat, not document retrieval.
    return normalized in greetings or len(words) <= 2


def needs_fallback(answer_text, retrieved_context=None):
    if not answer_text:
        return True

    normalized = answer_text.strip().lower()
    weak_phrases = [
        "don't have enough information",
        "do not have enough information",
        "based on the provided context",
        "provided context",
        "does not contain",
        "not available",
    ]

    if len(normalized) < 60:
        return True

    if not retrieved_context:
        return True

    return any(phrase in normalized for phrase in weak_phrases)


def clean_assistant_response(text):
    if not text:
        return "Sorry, I could not generate a response right now."

    cleaned = text.strip()
    # Hide reasoning tags if a model emits internal thoughts.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*(#{1,6}\s*)?(thinking|thought process|thought|reasoning)\s*:?\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned or "Sorry, I could not generate a response right now."


def get_config_value(key, default=""):
    value = os.environ.get(key)
    if value:
        return value
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def resolve_groq_model_name(raw_model_name):
    normalized = (raw_model_name or "").strip()
    if not normalized:
        return "llama-3.1-8b-instant"
    return DEPRECATED_GROQ_MODELS.get(normalized, normalized)


def build_user_friendly_error(exc):
    msg = str(exc).lower()
    if any(token in msg for token in ["api key", "authentication", "unauthorized", "401"]):
        return "Groq API key is missing or invalid. Set GROQ_API_KEY in Streamlit Secrets and redeploy."
    if "model" in msg and any(token in msg for token in ["not found", "does not exist", "unsupported", "decommissioned"]):
        return "The selected Groq model is unavailable/decommissioned. Set GROQ_MODEL_NAME to llama-3.1-8b-instant in Streamlit Secrets."
    if any(token in msg for token in ["rate limit", "quota", "429"]):
        return "Groq rate limit reached. Please wait a minute and try again."
    compact = str(exc).strip().replace("\n", " ")
    if len(compact) > 220:
        compact = compact[:220] + "..."
    return f"Backend request failed: {compact}"


def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .stApp {
                background: radial-gradient(circle at 10% 20%, #172554 0%, #0b1220 40%, #020617 100%);
                color: #e2e8f0;
            }

            .main .block-container {
                max-width: 775px;
                margin: 0 auto;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .hero-wrap {
                text-align: center;
                margin-bottom: 1rem;
            }

            .hero-title {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
                color: #f8fafc;
            }

            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1rem;
                margin-bottom: 1.2rem;
            }

            .hero-image-wrap {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(125, 211, 252, 0.35);
                box-shadow:
                    0 0 0 1px rgba(56, 189, 248, 0.2),
                    0 16px 40px rgba(2, 132, 199, 0.35),
                    0 0 60px rgba(6, 182, 212, 0.2);
                background: rgba(15, 23, 42, 0.4);
            }

            .hero-image-gap-top {
                margin-top: 0.35rem;
            }

            .hero-image-gap-bottom {
                margin-bottom: 1rem;
            }

            .welcome-card {
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 16px;
                padding: 1rem 1.2rem;
                background: rgba(15, 23, 42, 0.55);
                margin-bottom: 1rem;
                color: #e2e8f0;
            }

            .user-bubble,
            .assistant-bubble {
                border-radius: 16px;
                padding: 0.8rem 1rem;
                line-height: 1.5;
                margin-bottom: 0.25rem;
                border: 1px solid transparent;
            }

            .user-bubble {
                background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
                color: #eff6ff;
                border-color: rgba(191, 219, 254, 0.25);
            }

            .assistant-bubble {
                background: rgba(15, 23, 42, 0.8);
                color: #e2e8f0;
                border-color: rgba(148, 163, 184, 0.25);
            }

            .stButton > button {
                border-radius: 999px;
                border: 1px solid rgba(148, 163, 184, 0.35);
                background: rgba(15, 23, 42, 0.7);
                color: #e2e8f0;
                transition: all 0.2s ease;
            }

            .stButton > button:hover {
                border-color: rgba(147, 197, 253, 0.9);
                transform: translateY(-1px);
                background: rgba(30, 41, 59, 0.85);
            }

            .stChatInput textarea {
                border-radius: 16px !important;
            }

            section[data-testid="stSidebar"] {
                background: rgba(2, 6, 23, 0.85);
                border-right: 1px solid rgba(148, 163, 184, 0.2);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chat_bubble(role, content):
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    safe_content = html.escape(content).replace("\n", "<br>")
    st.chat_message(role).markdown(
        f"<div class='{bubble_class}'>{safe_content}</div>",
        unsafe_allow_html=True,
    )


def main():
    inject_custom_css()

    with st.sidebar:
        st.markdown("### 🩺 AI Medical Assistant")
        st.caption("Clinical Q&A assistant")
        if not get_config_value("GROQ_API_KEY", ""):
            st.warning("Missing GROQ_API_KEY in Streamlit Secrets.")
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">🩺 AI Medical Assistant</div>
            <div class="hero-subtitle">Ask health questions, explore medical topics, and get clear responses instantly.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'quick_prompt' not in st.session_state:
        st.session_state.quick_prompt = ""

    st.markdown("<div class='hero-image-gap-top'></div>", unsafe_allow_html=True)
    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
    with img_col2:
        st.markdown("<div class='hero-image-wrap'>", unsafe_allow_html=True)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(current_dir, "assets", "ai_doctor.png"),
            os.path.join(current_dir, "assests", "ai_doctor.png"),
        ]
        hero_image_path = next((p for p in candidate_paths if os.path.exists(p)), None)

        if hero_image_path:
            st.image(hero_image_path, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-image-gap-bottom'></div>", unsafe_allow_html=True)

    selected_prompt = None
    st.markdown("#### Quick Prompts")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👋 Hello", use_container_width=True):
            st.session_state.quick_prompt = "hello"
            selected_prompt = st.session_state.quick_prompt
        if st.button("🫀 What is hypertension?", use_container_width=True):
            st.session_state.quick_prompt = "What is hypertension?"
            selected_prompt = st.session_state.quick_prompt
    with col2:
        if st.button("🤒 Common fever causes", use_container_width=True):
            st.session_state.quick_prompt = "What are common causes of fever?"
            selected_prompt = st.session_state.quick_prompt
        if st.button("🥗 Healthy lifestyle tips", use_container_width=True):
            st.session_state.quick_prompt = "Give me healthy lifestyle tips."
            selected_prompt = st.session_state.quick_prompt

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                Welcome! I can answer general questions and medical-reference questions grounded in your documents.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        render_chat_bubble(message['role'], message['content'])

    prompt = st.chat_input("Ask anything about health, symptoms, or wellness...")
    if not prompt and selected_prompt:
        prompt = selected_prompt
    elif not prompt and st.session_state.quick_prompt:
        prompt = st.session_state.quick_prompt

    # Consume quick prompt once per click to avoid repeating on rerun.
    if prompt and prompt == st.session_state.quick_prompt:
        st.session_state.quick_prompt = ""

    if prompt:
        render_chat_bubble('user', prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
                
        try: 
            GROQ_API_KEY = get_config_value("GROQ_API_KEY", "")
            if not GROQ_API_KEY:
                result = "Groq API key is not configured. Add GROQ_API_KEY in Streamlit Secrets."
                render_chat_bubble('assistant', result)
                st.session_state.messages.append({'role':'assistant', 'content': result})
                return

            GROQ_MODEL_NAME = resolve_groq_model_name(get_config_value("GROQ_MODEL_NAME", "llama-3.1-8b-instant"))
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            with st.spinner("Thinking... 🤔"):
                if is_general_chat_query(prompt):
                    direct = llm.invoke(prompt)
                    result = getattr(direct, "content", str(direct)).strip()
                else:
                    vectorstore=get_vectorstore()
                    if vectorstore is None:
                        result = "Sorry, I could not load the medical knowledge base right now."
                    else:
                        retrieval_qa_chat_prompt = set_custom_prompt(RAG_PROMPT_TEMPLATE)

                        # Document combiner chain (stuff documents into prompt)
                        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

                        # Retrieval chain (retriever + doc combiner)
                        rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

                        response=rag_chain.invoke({'input': prompt})
                        result = response.get("answer", "")
                        retrieved_context = response.get("context", [])

                        if needs_fallback(result, retrieved_context):
                            fallback = llm.invoke(prompt)
                            result = getattr(fallback, "content", str(fallback)).strip()

            result = clean_assistant_response(result)
            render_chat_bubble('assistant', result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            logger.exception("Chat request failed")
            result = clean_assistant_response(build_user_friendly_error(e))
            render_chat_bubble('assistant', result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

if __name__ == "__main__":
    main() 