"""
Unified prompt management for RAG Document Chat system.
All prompts are defined here for easy debugging and adjustment.
"""

# System prompts
SYSTEM_PROMPT = """You are a professional document assistant specialized in helping users find and interpret information from uploaded PDF documents.

Your responsibilities:
1. Answer questions only based on the provided document context
2. If information is not in the documents, state this clearly
3. Cite sources for each reference (document name, page, location)
4. Use clear, structured formatting to organize answers
5. List all relevant information sources if multiple exist

Answer format requirements:
- Provide direct answer first
- Must mark each citation with [source]
- Source format: [document-Ppage-Sclause]
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
ANSWER_GENERATION_PROMPT = """Answer the question based on the following context information.

Question: {question}

Relevant document content (with source markers):
{context}

Please provide:
1. Core answer (2-3 sentences)
2. Detailed explanation (if needed)
3. Related supplementary information
4. List of cited sources

Answer:"""

# Citation prompt
CITATION_PROMPT = """Extract accurate source information from the following text.

Text: {text}
Source metadata:
- Document: {document}
- Page: {page}
- Clause: {chunk}

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
3. Check if documents are properly indexed (see "Total text clauses" on the left sidebar)
"""

# Collection/Sources formatting
SOURCES_HEADER = "📚 Information Sources:"

# Response templates
RESPONSE_TEMPLATE_SINGLE_SOURCE = """**Answer:** {answer}

**Source Information:**
- Document: {document}
- Page: {page}
- Clause: {chunk}
- Similarity: {score:.1%}
"""

RESPONSE_TEMPLATE_MULTIPLE_SOURCES = """**Answer:** {answer}

**Related Sources:**
{sources_list}
"""

SOURCE_ITEM_TEMPLATE = """- **{document}** (page {page}, clause {chunk}, similarity {score:.1%})
  > {excerpt}
"""

# Error messages
ERROR_OPENAI_API = "OpenAI API call failed, using extractive answer mode"
ERROR_INVALID_QUESTION = "Question cannot be empty. Please enter a valid question."
ERROR_NO_DOCUMENTS = "No documents have been indexed yet. Please upload PDF files first."

# Instructions for LLM
CITATION_INSTRUCTION = """Important: Your answer must include source citations.
Format: [document-Ppage-Sclause]
Example: [employee_handbook-P3-S2] According to the employee handbook page 3 clause 2...
"""

# Constrain response
LENGTH_CONSTRAINTS = {
    "short_answer": 100,  # characters
    "medium_answer": 300,  # characters
    "long_answer": 1000,  # characters
}
