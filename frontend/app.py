from __future__ import annotations

import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def setup_page() -> None:
    st.set_page_config(page_title="RAG Document Chat", page_icon="PDF", layout="wide")
    st.title("RAG Document Chat")
    st.caption("PDFs hochladen, indexieren und mit Quellenangaben durchsuchen.")


def api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def upload_pdfs(uploaded_files) -> list[str]:
    files = [
        ("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
        for uploaded_file in uploaded_files
    ]
    response = requests.post(api_url("/api/upload"), files=files, timeout=300)
    response.raise_for_status()
    return response.json()["messages"]


def get_chunk_count() -> int:
    response = requests.get(api_url("/api/documents/stats"), timeout=30)
    response.raise_for_status()
    return int(response.json()["chunk_count"])


def ask_question(question: str) -> dict:
    response = requests.post(api_url("/api/chat"), json={"question": question}, timeout=300)
    response.raise_for_status()
    return response.json()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Dokumente")
        st.caption(f"Backend: `{API_BASE_URL}`")
        uploaded_files = st.file_uploader(
            "PDFs hochladen",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("PDFs indexieren", type="primary", disabled=not uploaded_files):
            with st.spinner("PDFs werden gelesen, gechunked und eingebettet..."):
                try:
                    for message in upload_pdfs(uploaded_files):
                        st.write(message)
                except Exception as exc:
                    st.error(f"Indexierung fehlgeschlagen: {exc}")

        try:
            st.metric("Textabschnitte in ChromaDB", get_chunk_count())
        except Exception:
            st.warning("Backend nicht erreichbar.")

        if st.button("Chat leeren"):
            st.session_state.messages = []
            st.rerun()


def render_chat() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Frage zu den hochgeladenen PDFs stellen")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Suche relevante Textabschnitte..."):
            try:
                result = ask_question(question)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as exc:
                answer = f"Antwortgenerierung fehlgeschlagen: {exc}"
                sources = []
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Quellen anzeigen", expanded=False):
        for source in sources:
            score = source.get("score")
            score_text = f" | Treffer: {score:.2f}" if score is not None else ""
            st.markdown(
                f"**{source['document']}** | Seite {source['page']} | "
                f"Abschnitt {source['chunk']}{score_text}"
            )
            st.write(source["text"])


def main() -> None:
    setup_page()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
