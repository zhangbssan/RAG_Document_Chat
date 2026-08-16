from __future__ import annotations

import os
from typing import Generator

from openai import OpenAI, APIError

from app.config import OPENAI_MODEL,OPENAI_API_KEY
from app.rag import prompts
from app.schemas import Source


def format_context(sources: list[Source]) -> str:
    """Format sources into a context string for the LLM."""
    if not sources:
        return ""

    blocks = []
    for index, source in enumerate(sources, start=1):
        page_label = _page_label(source)
        source_ref = f"[{source.document}-P{page_label}-S{source.chunk}]"
        link_line = f"Link: {source.link}\n" if source.link else ""
        blocks.append(
            f"{source_ref}\n"
            f"Document: {source.document}\n"
            f"Page: {page_label}\n"
            f"{link_line}"
            f"Clause: {source.chunk}\n"
            f"Content: {source.text}"
        )
    return "\n\n".join(blocks)


def _page_label(source: Source) -> str:
    if source.pages and len(source.pages) > 1:
        return f"{source.pages[0]}-{source.pages[-1]}"
    return str(source.page)


def _validate_question(question: str) -> None:
    """Validate user question."""
    if not question or not question.strip():
        raise ValueError(prompts.ERROR_INVALID_QUESTION)


def answer_question(
    question: str,
    sources: list[Source],
    api_key: str | None = None,
) -> str:
    """
    Generate an answer to a question based on provided sources.
    Returns the complete answer as a string.
    """
    _validate_question(question)
    
    if not sources:
        return prompts.NO_SOURCES_FOUND_MESSAGE
    
    # Try to use OpenAI
    answer = answer_with_openai(question, sources, api_key=api_key)
    
    if answer:
        return answer
    
    # Fallback to extractive answer
    return fallback_answer(sources)


def answer_with_openai(
    question: str,
    sources: list[Source],
    api_key: str | None = None,
) -> str | None:
    """Generate answer using OpenAI API."""
    api_key = api_key or OPENAI_API_KEY
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)
        context = format_context(sources)
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompts.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompts.ANSWER_GENERATION_PROMPT.format(
                        question=question,
                        context=context,
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    except APIError as e:
        print(f"{prompts.ERROR_OPENAI_API}: {e}")
        return None


def answer_question_stream(
    question: str,
    sources: list[Source],
    api_key: str | None = None,
) -> Generator[str, None, None]:
    """
    Generate an answer to a question with streaming support.
    Yields text chunks as they are generated.
    """
    _validate_question(question)
    
    if not sources:
        yield prompts.NO_SOURCES_FOUND_MESSAGE
        return
    
    api_key = api_key or OPENAI_API_KEY
    if not api_key:
        # Fallback without streaming
        yield fallback_answer(sources)
        return
    
    try:
        client = OpenAI(api_key=api_key)
        context = format_context(sources)
        
        with client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompts.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompts.ANSWER_GENERATION_PROMPT.format(
                        question=question,
                        context=context,
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1000,
            stream=True,
        ) as response:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
    
    except APIError as e:
        print(f"{prompts.ERROR_OPENAI_API}: {e}")
        yield fallback_answer(sources)


def fallback_answer(sources: list[Source]) -> str:
    """Generate a fallback extractive answer when API is unavailable."""
    if not sources:
        return prompts.NO_SOURCES_FOUND_MESSAGE
    
    lines = [
        "**Extractive Answer (API unavailable):**\n",
        "Based on relevant content from the documents:\n",
    ]
    
    # Show top 3 sources with proper citations
    for i, source in enumerate(sources[:3], start=1):
        excerpt = source.text[:500].strip()
        if len(source.text) > 500:
            excerpt += "..."
        
        page_label = _page_label(source)
        citation = f"[{source.document}-P{page_label}-S{source.chunk}]"
        score_text = (
            f" (Rank score: {source.score:.4f})"
            if source.score is not None
            else ""
        )
        link_text = f" ({source.link})" if source.link else ""

        lines.append(
            f"\n**Source {i}:** {citation}{score_text}{link_text}\n"
            f">>> {excerpt}"
        )
    
    return "\n".join(lines)


def extract_sources_from_answer(answer: str, sources: list[Source]) -> list[Source]:
    """
    Extract source references from the generated answer.
    Returns top sources that were cited.
    """
    # Simple heuristic: return the top sources used
    return sources[:3] if sources else []
