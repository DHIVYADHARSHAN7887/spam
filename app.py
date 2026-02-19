import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(130,80,255,0.25);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Hero heading */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-sub {
    color: #9ca3af;
    font-size: 1rem;
    margin-top: 8px;
}

/* Result badges */
.badge-spam {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    box-shadow: 0 0 24px rgba(239,68,68,0.5);
    animation: pulse-red 1.5s infinite;
}
.badge-ham {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    box-shadow: 0 0 24px rgba(16,185,129,0.5);
    animation: pulse-green 1.5s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 18px rgba(239,68,68,0.5); }
    50%       { box-shadow: 0 0 36px rgba(239,68,68,0.8); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 18px rgba(16,185,129,0.5); }
    50%       { box-shadow: 0 0 36px rgba(16,185,129,0.8); }
}

/* Confidence bar */
.conf-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-top: 8px;
}
.conf-bar-fill-spam {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #fca5a5);
    border-radius: 8px;
    transition: width 0.6s ease;
}
.conf-bar-fill-ham {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #6ee7b7);
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* Section header */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #c4b5fd;
    border-left: 4px solid #7c3aed;
    padding-left: 12px;
    margin-bottom: 12px;
}

/* Streamlit overrides */
div[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading / Training ────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VEC_PATH   = os.path.join(MODEL_DIR, "vectorizer.pkl")
CSV_PATH   = os.path.join(MODEL_DIR, "spam.csv")


@st.cache_resource(show_spinner=False)
def load_or_train():
    """Load saved model/vectorizer or train from scratch."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        with open(VEC_PATH, "rb") as f:
            v = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        df = pd.read_csv(CSV_PATH)
        X_train, X_test, y_train, y_test = train_test_split(
            df.Message, df.Category, test_size=0.2, random_state=5
        )
        y_pred = model.predict(v.transform(X_test.values))
        acc = accuracy_score(y_test, y_pred)
        return v, model, df, y_test, y_pred, acc

    # Train from scratch
    df = pd.read_csv(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df.Message, df.Category, test_size=0.2, random_state=5
    )
    v = CountVectorizer()
    X_train_count = v.fit_transform(X_train.values)
    X_test_count  = v.transform(X_test.values)
    model = MultinomialNB()
    model.fit(X_train_count, y_train)
    y_pred = model.predict(X_test_count)
    acc = accuracy_score(y_test, y_pred)

    with open(VEC_PATH, "wb") as f:
        pickle.dump(v, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return v, model, df, y_test, y_pred, acc


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ **SpamShield AI**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Spam Detector", "📊 Model Analytics", "📋 Dataset Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem;color:#6b7280;'>
    <b>Model:</b> Naive Bayes (MultinomialNB)<br>
    <b>Vectorizer:</b> Count Vectorizer<br>
    <b>Dataset:</b> SMS Spam Collection<br>
    <b>Accuracy:</b> ~98.9%
    </div>
    """, unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading AI model…"):
    vectorizer, model, df, y_test, y_pred, accuracy = load_or_train()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SPAM DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
if "Spam Detector" in page:
    # Hero
    st.markdown('<div class="hero-title">🛡️ SpamShield AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-time spam detection powered by Naive Bayes + NLP</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Quick-stats row
    total  = len(df)
    n_spam = df[df.Category == "spam"].shape[0]
    n_ham  = df[df.Category == "ham"].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{accuracy*100:.1f}%", f"{total:,}", f"{n_spam:,}", f"{n_ham:,}"],
        ["Accuracy", "Total Samples", "Spam Messages", "Ham Messages"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">✍️ Enter a message to classify</div>', unsafe_allow_html=True)

    # Example messages
    examples = {
        "🤝 Normal message": "Hey, are we still meeting for lunch tomorrow?",
        "🏆 Spam example 1": "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "🎰 Spam example 2": "FREE entry! Win FA Cup Final tickets. Text FA to 87121. Cost 10p/min.",
        "📅 Normal example": "Can you pick up the kids from school today?",
    }

    sel = st.selectbox("💡 Try an example", ["— type your own —"] + list(examples.keys()))
    default_text = examples.get(sel, "")

    user_input = st.text_area(
        "Message",
        value=default_text,
        height=150,
        placeholder="Type or paste a message here…",
        label_visibility="collapsed",
    )

    if st.button("🔍 Analyze Message"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a message first.")
        else:
            vec = vectorizer.transform([user_input])
            prediction = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            spam_prob = proba[list(model.classes_).index("spam")]
            ham_prob  = proba[list(model.classes_).index("ham")]

            st.markdown("<br>", unsafe_allow_html=True)
            res_col, detail_col = st.columns([1, 2])

            with res_col:
                if prediction == "spam":
                    st.markdown('<div class="badge-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-ham">✅ LEGITIMATE (Ham)</div>', unsafe_allow_html=True)

            with detail_col:
                st.markdown(f"**🔴 Spam confidence:** {spam_prob*100:.1f}%")
                st.markdown(f"""<div class="conf-bar-wrap"><div class="conf-bar-fill-spam" style="width:{spam_prob*100:.1f}%"></div></div>""", unsafe_allow_html=True)

                st.markdown(f"**🟢 Ham confidence:** {ham_prob*100:.1f}%")
                st.markdown(f"""<div class="conf-bar-wrap"><div class="conf-bar-fill-ham" style="width:{ham_prob*100:.1f}%"></div></div>""", unsafe_allow_html=True)

    # ── Batch input ──────────────────────────────────────────────────────────
    with st.expander("📦 Batch Classify (one message per line)"):
        batch_text = st.text_area("Messages", height=120, placeholder="message 1\nmessage 2\n…", label_visibility="collapsed")
        if st.button("🚀 Classify All"):
            lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
            if lines:
                vecs = vectorizer.transform(lines)
                preds = model.predict(vecs)
                probas = model.predict_proba(vecs)
                sp_idx = list(model.classes_).index("spam")
                results = pd.DataFrame({
                    "Message": lines,
                    "Prediction": preds,
                    "Spam Probability": [f"{p[sp_idx]*100:.1f}%" for p in probas],
                })
                st.dataframe(results, use_container_width=True)
            else:
                st.info("Enter at least one message above.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="hero-title">📊 Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Performance metrics and visualisations for the trained Naive Bayes model</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)

    col_acc, col_prec, col_rec, col_f1 = st.columns(4)
    for col, key, label, emoji in zip(
        [col_acc, col_prec, col_rec, col_f1],
        ["accuracy", "weighted avg", "weighted avg", "weighted avg"],
        ["Accuracy", "Precision", "Recall", "F1-Score"],
        ["🎯", "🔬", "📡", "⚖️"],
    ):
        val = report["accuracy"] if key == "accuracy" else report["weighted avg"][{"Precision":"precision","Recall":"recall","F1-Score":"f1-score"}[label]]
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.8rem;">{emoji}</div>
            <div class="metric-value">{val*100:.1f}%</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    # Confusion matrix
    with left:
        st.markdown('<div class="section-title">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            cmap="RdPu", ax=ax,
            linewidths=0.5, linecolor="#2d2d4e",
        )
        ax.set_xlabel("Predicted", color="#c4b5fd", fontsize=11)
        ax.set_ylabel("Actual", color="#c4b5fd", fontsize=11)
        ax.tick_params(colors="#e5e7eb")
        plt.tight_layout()
        st.pyplot(fig)

    # Class distribution
    with right:
        st.markdown('<div class="section-title">🥧 Class Distribution</div>', unsafe_allow_html=True)
        counts = df["Category"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("#1a1a2e")
        ax2.set_facecolor("#1a1a2e")
        wedges, texts, autotexts = ax2.pie(
            counts.values,
            labels=counts.index.str.capitalize(),
            autopct="%1.1f%%",
            colors=["#10b981", "#ef4444"],
            startangle=140,
            wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
        )
        for t in texts + autotexts:
            t.set_color("#e5e7eb")
        plt.tight_layout()
        st.pyplot(fig2)

    # Per-class report table
    st.markdown('<div class="section-title">📋 Per-class Classification Report</div>', unsafe_allow_html=True)
    display_df = report_df.drop(index=["accuracy"], errors="ignore")
    st.dataframe(display_df.style.format("{:.3f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown('<div class="hero-title">📋 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Browse the SMS Spam Collection dataset used for training</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    f1, f2, f3 = st.columns([1, 2, 1])
    with f1:
        cat_filter = st.selectbox("Category", ["All", "Spam", "Ham"])
    with f2:
        search_query = st.text_input("🔎 Search messages", placeholder="keyword…", label_visibility="collapsed")
    with f3:
        n_rows = st.number_input("Show rows", min_value=5, max_value=500, value=20, step=5)

    filtered = df.copy()
    if cat_filter != "All":
        filtered = filtered[filtered.Category == cat_filter.lower()]
    if search_query:
        filtered = filtered[filtered.Message.str.contains(search_query, case=False, na=False)]

    st.markdown(f"**{len(filtered):,} messages** match your filters.")
    st.dataframe(filtered.head(n_rows), use_container_width=True, height=420)

    # Message length distribution
    st.markdown('<div class="section-title">📏 Message Length Distribution</div>', unsafe_allow_html=True)
    df_len = df.copy()
    df_len["length"] = df_len.Message.str.len()

    fig3, ax3 = plt.subplots(figsize=(9, 3))
    fig3.patch.set_facecolor("#1a1a2e")
    ax3.set_facecolor("#1a1a2e")
    for cat, color in [("ham", "#10b981"), ("spam", "#ef4444")]:
        subset = df_len[df_len.Category == cat]["length"]
        ax3.hist(subset, bins=50, alpha=0.7, color=color, label=cat.capitalize(), edgecolor="none")
    ax3.set_xlabel("Message Length (characters)", color="#9ca3af", fontsize=10)
    ax3.set_ylabel("Count", color="#9ca3af", fontsize=10)
    ax3.tick_params(colors="#9ca3af")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#374151")
    ax3.legend(facecolor="#1a1a2e", edgecolor="#374151", labelcolor="#e5e7eb")
    plt.tight_layout()
    st.pyplot(fig3)
