import re
import joblib
import streamlit as st

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Fallback
FALLBACK = "Sorry, I am not sure about that. I will pass you to a human agent."
GREETINGS = {"hello", "hi", "hey", "greetings", "good morning", "good evening", "salam", "aoa"}

# --- TEXT PREPROCESSING ---
def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\{\{.*?\}\}', ' slot ', text)
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    return ' '.join(text.split())

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_assets():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    replies = joblib.load("replies.pkl")
    return model, vectorizer, replies

try:
    model, vectorizer, REPLIES = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure 'logistic_model.pkl', 'tfidf_vectorizer.pkl', and 'replies.pkl' are placed in the app directory.")
    assets_loaded = False

# --- INTENT INFERENCE ENGINE ---
def get_bot_response(user_text: str, confidence_threshold: float = 0.45):
    cleaned = clean(user_text)

    # 1. Rule-based Greeting Interceptor
    first_word = cleaned.split()[0] if cleaned.split() else ""
    if cleaned in GREETINGS or first_word in GREETINGS:
        return (
            "Hello! I am your automated customer support assistant. How can I help you with your orders, accounts, or refunds today?",
            "greeting",
            1.0,
            [("greeting", 1.0)],
            True
        )

    # 2. Vector Transform & Out-of-Vocabulary Check
    features = vectorizer.transform([cleaned])
    if features.nnz == 0:
        return FALLBACK, "out_of_vocabulary", 0.0, [], False

    # 3. Model Prediction
    probabilities = model.predict_proba(features)[0]
    top_indices = probabilities.argsort()[::-1][:3]
    top_intents = [(model.classes_[i], float(probabilities[i])) for i in top_indices]

    best_intent, best_prob = top_intents[0]

    # 4. Confidence Threshold Check
    if best_prob < confidence_threshold:
        return FALLBACK, best_intent, best_prob, top_intents, False

    reply_content = REPLIES.get(best_intent, FALLBACK)
    return reply_content, best_intent, best_prob, top_intents, True

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ Engine Controls")
    
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.85,
        value=0.45,
        step=0.05,
        help="Inquiries with intent probability below this score will be routed to a human support agent."
    )
    
    show_diagnostics = st.toggle("Show Intent Diagnostics", value=True)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🤖 Customer Support Desk")
st.caption("Intent Classification & Routing Model (Bitext 27K)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

# Display historical messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_diagnostics and "meta" in message:
            with st.expander("🔍 Prediction Diagnostics"):
                st.write(f"**Predicted Intent:** `{message['meta']['intent']}`")
                st.write(f"**Confidence:** `{message['meta']['confidence']:.1%}`")
                st.write("**Top Candidates:**")
                for rank, (cand_intent, cand_score) in enumerate(message['meta']['top_k'], start=1):
                    st.write(f"{rank}. `{cand_intent}` — {cand_score:.1%}")

# Input prompt handler
if user_prompt := st.chat_input("Ask a question about your order, account, or refund..."):
    if not assets_loaded:
        st.stop()

    # Render User Message
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Run Prediction
    reply_text, intent, confidence, top_k, passed = get_bot_response(user_prompt, confidence_threshold=threshold)

    meta_info = {
        "intent": intent if passed else f"Fallback Triggered ({intent})",
        "confidence": confidence,
        "top_k": top_k
    }

    # Render Assistant Message
    with st.chat_message("assistant"):
        st.markdown(reply_text)
        if show_diagnostics:
            with st.expander("🔍 Prediction Diagnostics"):
                st.write(f"**Predicted Intent:** `{meta_info['intent']}`")
                st.write(f"**Confidence:** `{meta_info['confidence']:.1%}`")
                st.write("**Top Candidates:**")
                for rank, (cand_intent, cand_score) in enumerate(top_k, start=1):
                    st.write(f"{rank}. `{cand_intent}` — {cand_score:.1%}")

    # Persist Assistant State
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": reply_text,
        "meta": meta_info
    })