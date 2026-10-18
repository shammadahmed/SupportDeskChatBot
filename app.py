import re
import joblib
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Support Bot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

FALLBACK = "Sorry, I am not sure about that. I will pass you to a human agent."

# --- PREPROCESSING (Matches Kaggle notebook) ---
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\{\{.*?\}\}', ' slot ', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    return ' '.join(text.split())

# --- LOAD SAVED MODEL ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    replies = joblib.load("replies.pkl")
    return model, vectorizer, replies

try:
    model, vectorizer, REPLIES = load_artifacts()
    loaded = True
except FileNotFoundError as err:
    st.error(f"Missing pickle file: {err}. Ensure the 3 .pkl files are in the same directory as app.py.")
    loaded = False

# --- PREDICTION FUNCTION ---
def predict(message: str, threshold: float = 0.5):
    cleaned = clean(message)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    
    top_indices = probs.argsort()[::-1][:3]
    top_intents = [(model.classes_[i], float(probs[i])) for i in top_indices]
    
    best_intent, best_prob = top_intents[0]
    
    if best_prob < threshold:
        return FALLBACK, best_intent, best_prob, top_intents, False
    return REPLIES.get(best_intent, FALLBACK), best_intent, best_prob, top_intents, True

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Model Controls")
    
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Queries below this score route to human support fallback."
    )
    
    show_intent_stats = st.toggle("Show Intent Breakdown", value=True)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Support bot: Hello! How can I help?"}
        ]
        st.rerun()

# --- CHAT STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Support bot: Hello! How can I help?"}
    ]

st.title("💬 Customer Support Assistant")
st.caption("Trained on 27K Bitext Intent Dataset")

# Render previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_intent_stats and "debug" in msg:
            with st.expander("🔍 Intent Classification Details"):
                st.write(f"**Selected Intent:** `{msg['debug']['intent']}`")
                st.write(f"**Confidence:** `{msg['debug']['confidence']:.1%}`")
                st.write("**Top Predictions:**")
                for label, prob in msg["debug"]["top_intents"]:
                    st.write(f"- `{label}`: {prob:.1%}")

# --- HANDLE CHAT INPUT ---
if user_input := st.chat_input("Type your question (e.g., 'i want to cancel my order')..."):
    if not loaded:
        st.stop()
        
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Inference
    response, intent, score, top_k, matched = predict(user_input, threshold=threshold)
    
    debug_payload = {
        "intent": intent if matched else f"Handoff Fallback ({intent})",
        "confidence": score,
        "top_intents": top_k
    }

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
        if show_intent_stats:
            with st.expander("🔍 Intent Classification Details"):
                st.write(f"**Selected Intent:** `{debug_payload['intent']}`")
                st.write(f"**Confidence:** `{debug_payload['confidence']:.1%}`")
                st.write("**Top Predictions:**")
                for label, prob in top_k:
                    st.write(f"- `{label}`: {prob:.1%}")

    # Persist message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "debug": debug_payload
    })