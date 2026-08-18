from __future__ import annotations

import os
import json
import time
import uuid

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


def _chat_payload(
    question: str,
    conversation_id: str,
    openai_api_key: str | None = None,
) -> dict:
    """Build a chat request payload without persisting request-only secrets."""
    payload = {
        "question": question,
        "conversation_id": conversation_id,
        "top_k": 5,
    }
    if openai_api_key:
        payload["openai_api_key"] = openai_api_key
    return payload


def ask_question(
    question: str,
    conversation_id: str,
    openai_api_key: str | None = None,
) -> dict:
    """
    Ask a question via the agent-backed chat endpoint.

    Args:
        question: User question

    Returns:
        Response dict with answer and sources
    """
    response = requests.post(
        api_url("/api/chat"),
        json=_chat_payload(question, conversation_id, openai_api_key),
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def stream_answer(
    question: str,
    conversation_id: str,
    openai_api_key: str | None = None,
):
    """Stream Agent events for one persistent conversation turn."""
    try:
        response = requests.post(
            api_url("/api/chat/stream"),
            json=_chat_payload(question, conversation_id, openai_api_key),
            timeout=300,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        yield {"type": "error", "data": f"Stream processing failed: {str(exc)}"}


def run_evaluation() -> dict:
    """Run backend retrieval evaluation."""
    response = requests.post(api_url("/api/evaluate"), timeout=300)
    response.raise_for_status()
    return response.json()


def get_conversations() -> list[dict]:
    response = requests.get(api_url("/api/conversations"), timeout=30)
    response.raise_for_status()
    return response.json().get("conversations", [])


def get_conversation_history(conversation_id: str) -> dict:
    response = requests.get(
        api_url(f"/api/conversations/{conversation_id}/messages"),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def rename_chat_conversation(conversation_id: str, title: str) -> dict:
    response = requests.patch(
        api_url(f"/api/conversations/{conversation_id}"),
        json={"title": title},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def delete_chat_conversation(conversation_id: str) -> dict:
    response = requests.delete(
        api_url(f"/api/conversations/{conversation_id}"),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def start_new_conversation() -> None:
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.messages = []


def refresh_conversations() -> None:
    st.session_state.conversations = get_conversations()


def load_conversation(conversation_id: str) -> None:
    history = get_conversation_history(conversation_id)
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = history.get("messages", [])


def initialize_conversations() -> None:
    if "conversations" not in st.session_state:
        try:
            refresh_conversations()
        except Exception:
            st.session_state.conversations = []

    if "current_conversation_id" not in st.session_state:
        conversations = st.session_state.get("conversations", [])
        if conversations:
            try:
                load_conversation(conversations[0]["id"])
            except Exception:
                start_new_conversation()
        else:
            start_new_conversation()


# ==================== UI Components ====================


def render_sidebar() -> str | None:
    """Render sidebar with document management."""
    with st.sidebar:
        st.header("💬 Conversations")
        if st.button("＋ New chat", type="primary", use_container_width=True):
            start_new_conversation()
            st.rerun()

        conversations = st.session_state.get("conversations", [])
        current_id = st.session_state.get("current_conversation_id")
        if conversations:
            for conversation in conversations:
                conversation_id = conversation["id"]
                title = conversation.get("title") or "Untitled conversation"
                columns = st.columns([5, 1])
                with columns[0]:
                    label = f"● {title}" if conversation_id == current_id else title
                    if st.button(
                        label,
                        key=f"select_conversation_{conversation_id}",
                        use_container_width=True,
                    ):
                        try:
                            load_conversation(conversation_id)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to load conversation: {exc}")
                with columns[1]:
                    if st.button("×", key=f"delete_conversation_{conversation_id}"):
                        try:
                            delete_chat_conversation(conversation_id)
                            refresh_conversations()
                            if conversation_id == current_id:
                                remaining = st.session_state.conversations
                                if remaining:
                                    load_conversation(remaining[0]["id"])
                                else:
                                    start_new_conversation()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to delete conversation: {exc}")
        else:
            st.caption("Start a new chat to create your first saved conversation.")

        if current_id and any(item["id"] == current_id for item in conversations):
            with st.expander("Rename current chat"):
                current = next(item for item in conversations if item["id"] == current_id)
                new_title = st.text_input(
                    "Conversation title",
                    value=current.get("title", ""),
                    key=f"rename_title_{current_id}",
                )
                if st.button("Save title", key=f"rename_button_{current_id}"):
                    try:
                        rename_chat_conversation(current_id, new_title)
                        refresh_conversations()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to rename conversation: {exc}")

        st.divider()
        st.header("📁 Document Management")
        
        # Backend connection info
        st.caption(f"Backend: `{API_BASE_URL}`")
        openai_api_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help=(
                "Used only for this session/request. If empty, the app falls back "
                "to backend env key or extractive answers."
            ),
        ).strip()
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
                st.metric("Total Text Chunks", document_list.get("total_chunks", 0))
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
                            f"📄 {document_name} | Pages: {pages} | Chunks: {chunks}"
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
                                    f"({result.get('deleted_chunks', 0)} chunks)."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"❌ Delete failed: {exc}")
            else:
                st.info("No documents indexed yet")
        
        except Exception as e:
            st.warning(f"⚠️ Failed to get statistics: {e}")
        
        st.divider()
        
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

        return openai_api_key or None


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
                pages = source.get("pages")
                page_label = f"{pages[0]}-{pages[-1]}" if pages and len(pages) > 1 else source['page']
                st.markdown(
                    f"**Source {i}:** {source['document']} | "
                    f"Page {page_label} | "
                    f"Chunk {source['chunk']}"
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
            if source.get("link"):
                st.caption(f"🔗 {source['link']}")
            st.divider()


def render_chat(openai_api_key: str | None = None) -> None:
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
            status_placeholder = st.empty()
            
            try:
                full_response = ""
                sources = []
                with st.spinner("⏳ Thinking..."):
                    for event in stream_answer(
                        prompt,
                        conversation_id=st.session_state.current_conversation_id,
                        openai_api_key=openai_api_key,
                    ):
                        event_type = event.get("type")
                        if event_type == "tool_start":
                            query = event.get("query", "")
                            status_placeholder.info(f"🔎 Searching uploaded documents: {query}")
                        elif event_type == "sources":
                            sources = event.get("data", [])
                            status_placeholder.success(f"Found {len(sources)} source block(s)")
                        elif event_type == "token":
                            full_response += event.get("data", "")
                            message_placeholder.markdown(full_response + "▌")
                        elif event_type == "notice":
                            status_placeholder.warning(event.get("data", ""))
                        elif event_type == "error":
                            full_response = f"❌ {event.get('data', 'Agent request failed')}"
                            status_placeholder.empty()
                        elif event_type == "done":
                            status_placeholder.empty()

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
                try:
                    refresh_conversations()
                except Exception:
                    pass
            
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
    initialize_conversations()
    openai_api_key = render_sidebar()
    chat_tab, evaluation_tab = st.tabs(["Chat", "Evaluation"])
    with chat_tab:
        render_chat(openai_api_key=openai_api_key)
    with evaluation_tab:
        render_evaluation_panel()


if __name__ == "__main__":
    main()
