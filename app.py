
import tempfile
from pathlib import Path

import requests
import streamlit as st

from rag_pipeline import (
    collection,
    index_pdf,
    retrieve_context,
    generate_answer,
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind | AI Knowledge Studio",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #080b14;
        --text: #f2f4ff;
        --muted: #929bb7;
        --purple: #9b7bff;
        --cyan: #60e7f5;
        --border: rgba(164, 176, 230, 0.14);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 4%, rgba(112,72,220,.17), transparent 35%),
            radial-gradient(ellipse at 88% 25%, rgba(36,184,208,.10), transparent 30%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8,11,20,.55);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0e1220,#0a0d17);
        border-right: 1px solid var(--border);
    }

    h1,h2,h3,h4 {
        font-family: 'Space Grotesk',sans-serif !important;
        color: var(--text) !important;
    }

    p,li,label {
        color: #d2d8ee;
    }

    .ambient {
        position: fixed;
        top: 13%;
        right: 4%;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(128,87,255,.13);
        filter: blur(85px);
        pointer-events: none;
        z-index: 0;
        animation: glow 9s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from {transform: translate(0,0) scale(.9);}
        to {transform: translate(-45px,40px) scale(1.2);}
    }

    @keyframes rise {
        from {opacity:0; transform:translateY(14px);}
        to {opacity:1; transform:translateY(0);}
    }

    .brand {
        display:flex;
        align-items:center;
        gap:11px;
        margin-bottom:12px;
    }

    .brand-mark {
        width:42px;
        height:42px;
        display:grid;
        place-items:center;
        border-radius:14px;
        color:white;
        font-size:23px;
        background:linear-gradient(135deg,#a17bff,#5a61f6 60%,#4ddce9);
        box-shadow:0 0 28px rgba(132,100,255,.28);
    }

    .brand-name {
        font-family:'Space Grotesk',sans-serif;
        font-size:23px;
        font-weight:700;
        color:#f5f5ff;
    }

    .brand-name span {color:#a991ff;}

    .eyebrow {
        color:#9d8aff;
        font-size:11px;
        font-weight:700;
        text-transform:uppercase;
        letter-spacing:2.4px;
        margin-bottom:12px;
    }

    .hero {
        padding:35px 36px 31px;
        border:1px solid rgba(168,151,255,.22);
        border-radius:25px;
        background:
            radial-gradient(ellipse at 90% 0%,rgba(111,85,255,.23),transparent 40%),
            linear-gradient(130deg,rgba(24,30,53,.97),rgba(13,18,32,.96));
        box-shadow:0 18px 65px rgba(0,0,0,.19);
        animation:rise .65s ease both;
    }

    .hero h1 {
        font-size:clamp(32px,4.5vw,52px) !important;
        line-height:1.1;
        letter-spacing:-2px !important;
        margin:0 0 13px;
    }

    .gradient-text {
        background:linear-gradient(100deg,#c2aaff,#8c9aff 50%,#66e6f0);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
    }

    .hero-copy {
        max-width:650px;
        font-size:15px;
        line-height:1.8;
        color:#aeb8d4;
        margin-bottom:23px;
    }

    .status-pill {
        display:inline-flex;
        align-items:center;
        gap:9px;
        padding:8px 13px;
        border:1px solid rgba(96,231,245,.19);
        border-radius:99px;
        background:rgba(51,190,203,.07);
        color:#b4f4f6;
        font-size:12px;
        font-weight:600;
    }

    .status-dot {
        width:7px;
        height:7px;
        border-radius:50%;
        background:#60e7c1;
    }

    div[data-testid="stMetric"] {
        background:linear-gradient(145deg,rgba(22,28,47,.94),rgba(13,18,31,.94));
        border:1px solid var(--border);
        padding:17px 19px;
        border-radius:17px;
    }

    div[data-testid="stMetricLabel"] {
        color:#9ca8c7 !important;
    }

    div[data-testid="stMetricValue"] {
        color:#f5f5ff !important;
        font-family:'Space Grotesk',sans-serif;
    }

    .stButton > button {
        border-radius:12px;
        border:1px solid rgba(161,137,255,.36);
        background:linear-gradient(110deg,#8061f5,#625fe9);
        color:white;
        font-weight:700;
        padding:.63rem 1rem;
        transition:transform .2s ease,box-shadow .2s ease;
    }

    .stButton > button:hover {
        color:white;
        transform:translateY(-2px);
        box-shadow:0 8px 25px rgba(121,93,255,.23);
        border-color:#a997ff;
    }

    [data-testid="stChatInput"] {
        border-color:rgba(155,123,255,.25) !important;
        background:rgba(15,20,35,.96) !important;
        border-radius:17px !important;
    }

    [data-testid="stChatMessage"] {
        border:1px solid rgba(156,169,218,.11);
        border-radius:18px;
        padding:15px 17px;
        margin-bottom:13px;
        background:rgba(17,23,39,.74);
        animation:rise .35s ease both;
    }

    [data-testid="stExpander"] {
        border:1px solid rgba(151,166,216,.15);
        border-radius:13px;
        background:rgba(17,23,39,.65);
    }

    .sidebar-heading {
        color:#9c8bff;
        font-family:'Space Grotesk',sans-serif;
        font-size:12px;
        text-transform:uppercase;
        letter-spacing:1.8px;
        font-weight:700;
        margin:25px 0 12px;
    }

    .side-note {
        color:#8995b3;
        font-size:12px;
        line-height:1.7;
    }

    .feature-chip {
        display:inline-block;
        border:1px solid rgba(151,125,255,.2);
        background:rgba(133,102,255,.08);
        color:#c6b8ff;
        border-radius:8px;
        padding:6px 9px;
        font-size:11px;
        margin:3px 4px 3px 0;
    }

    .section-title {
        font-family:'Space Grotesk',sans-serif;
        font-size:21px;
        font-weight:700;
        color:#f0f2ff;
        margin:27px 0 5px;
    }

    .section-subtitle {
        color:#8f9ab7;
        font-size:13px;
        margin-bottom:17px;
    }

    .welcome-card {
        text-align:center;
        padding:42px 22px;
        margin:12px 0 20px;
        border:1px solid rgba(151,125,255,.18);
        border-radius:22px;
        background:
            radial-gradient(ellipse at 50% 0%,rgba(126,93,255,.12),transparent 65%),
            rgba(16,21,36,.72);
    }

    .welcome-icon {
        font-size:38px;
        margin-bottom:12px;
    }

    .welcome-title {
        font-family:'Space Grotesk',sans-serif;
        color:#f2f3ff;
        font-size:24px;
        font-weight:700;
        margin-bottom:9px;
    }

    .welcome-copy {
        max-width:490px;
        margin:auto;
        color:#96a2c0;
        font-size:13px;
        line-height:1.8;
    }

    .footer {
        text-align:center;
        color:#65718e;
        font-size:11px;
        padding:25px 0 12px;
        letter-spacing:.5px;
    }

    @media (max-width:700px) {
        .hero {padding:25px 20px;border-radius:19px;}
        .hero h1 {letter-spacing:-1px !important;}
    }

    @media (prefers-reduced-motion:reduce) {
        *,*::before,*::after {
            animation-duration:.01ms !important;
            animation-iteration-count:1 !important;
            transition-duration:.01ms !important;
        }
    }
    </style>

    <div class="ambient"></div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✳</div>
            <div class="brand-name">Docu<span>Mind</span></div>
        </div>
        <div class="side-note">
            YOUR PRIVATE AI KNOWLEDGE STUDIO<br>
            Turn documents into answers.
        </div>
        <div class="sidebar-heading">01 / Knowledge base</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload one PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if st.button(
        "✦  Index document",
        type="primary",
        use_container_width=True,
    ):

        if uploaded_file is None:
            st.warning("Upload a PDF first.")

        else:

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = Path(temp_file.name)

                with st.spinner("Reading and indexing your PDF..."):

                    chunk_count = index_pdf(
                        temp_path,
                        source_name=uploaded_file.name
                    )

                st.session_state.messages = []

                st.success(
                    f"Indexed {chunk_count} chunks from "
                    f"{uploaded_file.name}."
                )

                st.rerun()

            except Exception as error:

                st.error(f"Indexing failed: {error}")

            finally:

                if temp_path and temp_path.exists():
                    temp_path.unlink()

    st.markdown(
        '<div class="sidebar-heading">02 / System status</div>',
        unsafe_allow_html=True
    )

    st.metric(
        "Knowledge chunks",
        collection.count()
    )

    st.markdown(
        """
        <div class="sidebar-heading">03 / Capabilities</div>
        <span class="feature-chip">PDF intelligence</span>
        <span class="feature-chip">Semantic search</span>
        <span class="feature-chip">Source citations</span>
        <span class="feature-chip">Local AI</span>
        <br><br>
        <div class="side-note">
            Answers are generated using your indexed PDF
            and your local Ollama model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(
        "↺  Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div class="footer">
            DOCUMIND STUDIO · BUILT FOR KNOWLEDGE
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 4. MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ Intelligent document workspace</div>
        <h1>Your knowledge.<br>
        <span class="gradient-text">One conversation away.</span></h1>
        <div class="hero-copy">
            Search your PDFs by meaning, ask questions in natural language,
            and explore the evidence behind every answer — powered by your
            own retrieval-augmented generation pipeline.
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            Private document workspace
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 5. DASHBOARD METRICS
# --------------------------------------------------

st.markdown("")

metric_a, metric_b, metric_c = st.columns(3)

with metric_a:
    st.metric("Indexed knowledge", collection.count())

with metric_b:
    st.metric("AI workflow", "RAG")

with metric_c:
    st.metric("Answer evidence", "Sources")

st.markdown(
    """
    <div class="section-title">✧ Conversation studio</div>
    <div class="section-subtitle">
        Ask questions about your indexed PDF and inspect the evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 6. CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-icon">✳</div>
            <div class="welcome-title">
                Let's explore your documents.
            </div>
            <div class="welcome-copy">
                Upload and index a PDF from the left panel,
                then ask a question below.
                DocuMind retrieves document passages and
                generates answers grounded in your PDF.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        for index, source in enumerate(
            message.get("sources", []),
            start=1
        ):

            with st.expander(
                f"Source {index} · {source['source']} · Page {source['page']}"
            ):

                st.write(source["text"])

                st.caption(
                    f"Cosine distance: {source['distance']:.4f}"
                )


# --------------------------------------------------
# 7. ASK DOCUMIND
# --------------------------------------------------

question = st.chat_input(
    "Ask anything about your indexed PDF..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": []
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            if collection.count() == 0:

                answer = (
                    "Your knowledge base is empty. "
                    "Upload and index a PDF first."
                )

                sources = []

            else:

                # Retrieve every indexed chunk.
                with st.spinner(
                    "Reading your complete document..."
                ):

                    sources = retrieve_context(
                        question,
                        top_k=collection.count(),
                        max_distance=None
                    )

                # Generate answer from retrieved context.
                with st.spinner(
                    "Generating an evidence-based answer..."
                ):

                    answer = generate_answer(
                        question,
                        sources
                    )

            st.markdown(answer)

            if sources:

                st.markdown(
                    "#### ✧ Evidence from your documents"
                )

                for index, source in enumerate(
                    sources,
                    start=1
                ):

                    with st.expander(
                        f"Source {index} · {source['source']} · Page {source['page']}"
                    ):

                        st.write(source["text"])

                        st.caption(
                            f"Cosine distance: {source['distance']:.4f}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except requests.exceptions.ConnectionError:

            error_message = (
                "Cannot connect to Ollama. "
                "Please make sure Ollama is running."
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })

        except requests.exceptions.Timeout:

            error_message = (
                "Ollama took too long to respond. "
                "Try again with a shorter question."
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })

        except Exception as error:

            error_message = f"Something went wrong: {error}"

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        DOCUMIND · RETRIEVE WITH PRECISION · ANSWER WITH CONTEXT
    </div>
    """,
    unsafe_allow_html=True,
)
