"""
Streamlit Chat UI for Qwen2.5-3B-Instruct inference.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_config import MODEL_PATH_OR_ID, MODEL_KWARGS, TOKENIZER_KWARGS

# ── Page Config ──────────────────────────────────
st.set_page_config(
    page_title="AIq - Qwen Chat",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .stChatMessage {
        border-radius: 12px !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model Loading (cached) ──────────────────────
@st.cache_resource(show_spinner="Loading model... This may take a minute on first run.")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_OR_ID, **TOKENIZER_KWARGS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_OR_ID, **MODEL_KWARGS)
    model.eval()
    return model, tokenizer


# ── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Generation Settings")
    max_new_tokens = st.slider("Max new tokens", 32, 1024, 256, step=32)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    top_k = st.slider("Top-k", 1, 100, 50)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, step=0.05)

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.markdown(f"**Model:** `{MODEL_PATH_OR_ID.split('/')[-1] if '/' in MODEL_PATH_OR_ID else Path(MODEL_PATH_OR_ID).name}`")
    st.markdown(f"**Device:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Header ───────────────────────────────────────
st.markdown('<div class="main-header">AIq Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Qwen2.5-3B-Instruct</div>', unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────
model, tokenizer = load_model()


# ── Chat History ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat Input ───────────────────────────────────
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation for Qwen chat template
    chat_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_text = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
