"""
Unified prompt management for RAG Document Chat system.
All prompts are defined here for easy debugging and adjustment.
"""

# System prompts
SYSTEM_PROMPT = """You are a document-grounded assistant for uploaded PDF content.

Strict grounding rules:
1. Answer only from the provided document context.
2. Do not use outside knowledge, assumptions, or general background information.
3. Do not infer facts that are not explicitly supported by the context.
4. If the context is insufficient, say the uploaded documents do not contain enough information to answer.
5. Cite every factual claim with one of the provided source markers.
6. Do not invent document names, page numbers, chunks, or citations.
7. Keep answers concise, structured, and limited to what the retrieved context supports.

Citation format:
- Use only source markers that appear in the context.
- Source marker format: [document-Ppage-Schunk]
"""

# Question analysis prompt
QUESTION_ANALYSIS_PROMPT = """Analyze the user question and extract key information:

Question: {question}

Please identify:
1. Main topic
2. Key entities
3. Question type (fact, explanation, comparison, etc.)
4. Relevant timeline (if applicable)

Analysis results (concise format):"""

# Answer generation prompt
ANSWER_GENERATION_PROMPT = """Answer the question using only the provided document context.

Question: {question}

Relevant document content (with source markers):
{context}

Instructions:
- Do not use outside knowledge.
- Do not assume or add facts not present in the context.
- Cite every factual claim with source markers copied exactly from the context.
- If the context does not contain enough information, clearly state that the uploaded documents do not contain enough information to answer.
- Do not invent citations, document names, page numbers, or chunks.
- Keep the answer concise and structured.

Suggested format:
1. Direct answer with citations
2. Supporting details, only if useful
3. Insufficient information, if applicable

Answer:"""

# Citation prompt
CITATION_PROMPT = """Extract accurate source information from the following text.

Text: {text}
Source metadata:
- Document: {document}
- Page: {page}
- Chunk: {chunk}

Generate standardized citation format: [{document}-P{page}-S{chunk}]"""

# Fallback response when no sources found
NO_SOURCES_FOUND_MESSAGE = """No relevant information found in the indexed documents.

Possible reasons:
- Your question is not addressed in the uploaded PDFs
- More relevant documents need to be uploaded
- The question wording may need adjustment

Suggestions:
1. Try rephrasing your question with different keywords
2. Upload other PDF files containing relevant information
3. Check if documents are properly indexed (see "Total Text Chunks" on the left sidebar)
"""

# Collection/Sources formatting
SOURCES_HEADER = "📚 Information Sources:"

# Response templates
RESPONSE_TEMPLATE_SINGLE_SOURCE = """**Answer:** {answer}

**Source Information:**
- Document: {document}
- Page: {page}
- Chunk: {chunk}
- Rank score: {score:.4f}
"""

RESPONSE_TEMPLATE_MULTIPLE_SOURCES = """**Answer:** {answer}

**Related Sources:**
{sources_list}
"""

SOURCE_ITEM_TEMPLATE = """- **{document}** (page {page}, chunk {chunk}, rank score {score:.4f})
  > {excerpt}
"""

# Error messages
ERROR_OPENAI_API = "OpenAI API call failed, using extractive answer mode"
ERROR_INVALID_QUESTION = "Question cannot be empty. Please enter a valid question."
ERROR_NO_DOCUMENTS = "No documents have been indexed yet. Please upload PDF files first."

# Instructions for LLM
CITATION_INSTRUCTION = """Important: Every factual claim must include a source citation copied from the provided context.
Use only existing source markers. Do not invent document names, page numbers, chunks, or citations.
Format: [document-Ppage-Schunk]
If the provided context is insufficient, say the uploaded documents do not contain enough information to answer.
"""

# Constrain response
LENGTH_CONSTRAINTS = {
    "short_answer": 100,  # characters
    "medium_answer": 300,  # characters
    "long_answer": 1000,  # characters
}
