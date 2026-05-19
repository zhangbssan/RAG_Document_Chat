from __future__ import annotations

import os
import json
import time

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
EVALUATION_SAMPLE_DOCUMENTS = [
    "employee_handbook_en.pdf",
    "product_manual_en.pdf",
    "service_agreement_en.pdf",
]


# ==================== Page Setup ====================


def setup_page() -> None:
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="RAG Document Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📚 RAG Document Chat")
    st.caption("Upload PDFs, chat intelligently with document content, get answers with source citations")


# ==================== API Helpers ====================


def api_url(path: str) -> str:
    """Build full API URL."""
    return f"{API_BASE_URL.rstrip('/')}{path}"


def upload_pdfs(uploaded_files) -> list[str]:
    """Upload PDF files to the backend."""
    files = [
        ("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
        for uploaded_file in uploaded_files
    ]
    response = requests.post(api_url("/api/upload"), files=files, timeout=300)
    response.raise_for_status()
    return response.json()["messages"]


def get_document_list() -> dict:
    """Get indexed document details from backend."""
    response = requests.get(api_url("/api/documents/list"), timeout=30)
    response.raise_for_status()
    return response.json()


def delete_document(file_hash: str) -> dict:
    """Delete one indexed document from the backend."""
    response = requests.delete(api_url(f"/api/documents/{file_hash}"), timeout=60)
    response.raise_for_status()
    return response.json()


def ask_question(question: str, use_stream: bool = False) -> dict | None:
    """
    Ask a question.
    
    Args:
        question: User question
        use_stream: Whether to use streaming endpoint
        
    Returns:
        Response dict with answer and sources, or None if streaming
    """
    if use_stream:
        return None  # Streaming handled separately
    
    response = requests.post(
        api_url("/api/chat"),
        json={"question": question},
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def run_evaluation() -> dict:
    """Run backend retrieval evaluation."""
    response = requests.post(api_url("/api/evaluate"), timeout=300)
    response.raise_for_status()
    return response.json()


def stream_answer(question: str):
    """Stream answer chunks from the backend."""
    try:
        response = requests.post(
            api_url("/api/chat/stream"),
            json={"question": question},
            timeout=300,
            stream=True,
        )
        response.raise_for_status()
        
        sources = []
        
        for line in response.iter_lines():
            if not line:
                continue

            # New backend format: application/x-ndjson
            data_line = line

            # Backward compatibility for the previous SSE-style stream.
            if line.startswith(b"data: "):
                data_line = line[6:]

            try:
                data = json.loads(data_line.decode("utf-8"))

                if data.get("type") == "sources":
                    sources = data.get("data", [])
                elif data.get("type") in {"token", "content"}:
                    yield ("content", data.get("data", ""))
                elif data.get("type") == "done":
                    yield ("done", None)
                    yield ("sources", sources)

            except json.JSONDecodeError:
                continue
    
    except Exception as e:
        yield ("error", f"Stream processing failed: {str(e)}")


# ==================== UI Components ====================


def render_sidebar() -> None:
    """Render sidebar with document management."""
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Backend connection info
        st.caption(f"Backend: `{API_BASE_URL}`")
        if st.session_state.get("document_delete_message"):
            st.success(st.session_state.pop("document_delete_message"))
        
        # File uploader
        st.subheader("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Select PDF files to upload",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files. Maximum size: 25 MB per file.",
            label_visibility="collapsed",
        )
        
        if st.button(
            "🚀 Start Indexing",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        ):
            with st.spinner("📖 Processing PDF files..."):
                try:
                    messages = upload_pdfs(uploaded_files)
                    for message in messages:
                        st.success(message)
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Indexing failed: {exc}")
        
        st.divider()
        
        # Document statistics
        st.subheader("📊 Index Statistics")
        try:
            document_list = get_document_list()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Text Clauses", document_list.get("total_chunks", 0))
            with col2:
                st.metric("Indexed Documents", document_list.get("total_documents", 0))
            
            # List documents
            documents = document_list.get("documents", [])
            if documents:
                st.subheader("Indexed Documents")
                for doc in documents:
                    document_name = doc.get("document_name", "Unknown")
                    file_hash = doc.get("file_hash")
                    pages = doc.get("pages", 0)
                    chunks = doc.get("chunks", 0)

                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.caption(
                            f"📄 {document_name} | Pages: {pages} | Clauses: {chunks}"
                        )
                    with cols[1]:
                        if st.button(
                            "Delete",
                            key=f"delete_{file_hash}",
                            disabled=not file_hash,
                            use_container_width=True,
                        ):
                            try:
                                result = delete_document(file_hash)
                                st.session_state.document_delete_message = (
                                    f"Deleted {result.get('document_name', document_name)} "
                                    f"({result.get('deleted_chunks', 0)} clauses)."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"❌ Delete failed: {exc}")
            else:
                st.info("No documents indexed yet")
        
        except Exception as e:
            st.warning(f"⚠️ Failed to get statistics: {e}")
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # About section
        st.divider()
        st.markdown(
            """
            ### 💡 Instructions
            
            1. **Upload Documents**: Select PDF files and click "Start Indexing"
            2. **Ask Questions**: Enter your question in the input box below
            3. **View Sources**: Each answer displays referenced documents and page numbers
            4. **Multi-turn Chat**: You can perform multiple rounds of conversation
            """
        )


def render_sources(sources: list[dict], expandable: bool = True) -> None:
    """Render source citations for an answer."""
    if not sources:
        return

    if expandable:
        with st.expander("📖 View Sources", expanded=False):
            render_source_items(sources)
    else:
        render_source_items(sources)


def render_source_items(sources: list[dict]) -> None:
    """Render source rows without adding a wrapper container."""
    for i, source in enumerate(sources, 1):
        with st.container():
            # Source header
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(
                    f"**Source {i}:** {source['document']} | "
                    f"Page {source['page']} | "
                    f"Clause {source['chunk']}"
                )
            with cols[1]:
                if source.get("score") is not None:
                    score = source.get("score", 0)
                    st.caption(f"Retrieval rank score: {score:.4f}")

            # Source content
            st.markdown(
                f"> {source['text'][:400]}"
                + ("..." if len(source['text']) > 400 else "")
            )
            st.divider()


def render_chat() -> None:
    """Render chat interface."""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("💬 Enter your question..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            try:
                # Stream the response
                full_response = ""
                sources = []
                
                with st.spinner("⏳ Thinking..."):
                    for event_type, event_data in stream_answer(prompt):
                        if event_type == "content":
                            full_response += event_data
                            message_placeholder.markdown(full_response + "▌")
                        elif event_type == "sources":
                            sources = event_data
                        elif event_type == "error":
                            # Fallback: use regular endpoint if streaming fails
                            try:
                                result = ask_question(prompt, use_stream=False)
                                full_response = result["answer"]
                                sources = result.get("sources", [])
                            except Exception as fallback_error:
                                full_response = f"❌ Error: {event_data}"
                
                # Final render without cursor
                message_placeholder.markdown(full_response)
                
                # Display sources if any
                if sources:
                    with sources_placeholder:
                        render_sources(sources)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })
            
            except Exception as exc:
                error_message = f"❌ Request processing failed: {str(exc)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })


def render_evaluation_panel() -> None:
    """Render retrieval evaluation results."""
    st.subheader("Evaluation Panel")
    st.caption("Evaluation uses 5 predefined questions for the sample PDFs.")

    evaluation = st.session_state.get("evaluation_results")
    required_documents = (
        evaluation.get("required_documents", []) if evaluation else EVALUATION_SAMPLE_DOCUMENTS
    )
    indexed_documents = evaluation.get("indexed_documents", []) if evaluation else []
    missing_documents = evaluation.get("missing_documents", []) if evaluation else []

    if not evaluation:
        try:
            document_list = get_document_list()
            indexed_documents = sorted(
                doc.get("document_name", "")
                for doc in document_list.get("documents", [])
                if doc.get("document_name")
            )
            missing_documents = sorted(
                set(required_documents) - set(indexed_documents)
            )
        except Exception:
            indexed_documents = []
            missing_documents = []

    if required_documents:
        st.markdown("**Required sample documents:**")
        for document in required_documents:
            st.caption(f"`sample_docs/{document}`")

    if indexed_documents:
        with st.expander("Currently indexed documents", expanded=False):
            for document in indexed_documents:
                st.caption(document)

    if missing_documents:
        st.warning(
            "Some required sample documents are missing. Please upload them before "
            "running evaluation. Scores may be low.\n\n"
            "Missing sample documents:\n"
            + "\n".join(
                f"- `sample_docs/{document}`" for document in missing_documents
            )
        )

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                evaluation = run_evaluation()
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                return

        if evaluation.get("status") == "error":
            st.error(evaluation.get("detail", "Evaluation failed."))
            return

        st.session_state.evaluation_results = evaluation
        st.rerun()

    evaluation = st.session_state.get("evaluation_results")
    if not evaluation:
        st.info("Run evaluation to see retrieval scores.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tests Run", evaluation.get("tests_run", 0))
    with col2:
        st.metric(
            "Average Final Score",
            f"{evaluation.get('average_final_score', 0):.2f}",
        )

    for result in evaluation.get("results", []):
        document_hit_score = result.get(
            "document_hit_score",
            result.get("source_hit_score", 0.0),
        )
        page_hit_score = result.get("page_hit_score", 0.0)
        keyword_score = result.get("keyword_score", 0.0)
        final_score = result.get("final_score", 0.0)
        title = (
            f"{result['id']} | Final {final_score:.2f} | "
            f"Document {document_hit_score:.2f} | "
            f"Page {page_hit_score:.2f} | "
            f"Keywords {keyword_score:.2f}"
        )
        with st.expander(title, expanded=False):
            st.markdown(f"**Question:** {result['question']}")
            st.markdown(f"**Expected Answer:** {result['expected_answer']}")
            st.markdown(
                "**Expected Source:** "
                f"{result['expected_document']} page {result['expected_page']}"
            )
            st.markdown(
                "**Expected Keywords:** "
                + ", ".join(result.get("expected_keywords", []))
            )
            score_cols = st.columns(4)
            with score_cols[0]:
                st.metric("Document Hit", f"{document_hit_score:.2f}")
            with score_cols[1]:
                st.metric("Page Hit", f"{page_hit_score:.2f}")
            with score_cols[2]:
                st.metric("Keyword Score", f"{keyword_score:.2f}")
            with score_cols[3]:
                st.metric("Final Score", f"{final_score:.2f}")

            sources = result.get("retrieved_sources", [])
            if sources:
                render_sources(sources, expandable=False)
            else:
                st.warning("No sources retrieved.")


# ==================== Main ====================


def main() -> None:
    """Main application entry point."""
    setup_page()
    render_sidebar()
    chat_tab, evaluation_tab = st.tabs(["Chat", "Evaluation"])
    with chat_tab:
        render_chat()
    with evaluation_tab:
        render_evaluation_panel()


if __name__ == "__main__":
    main()
