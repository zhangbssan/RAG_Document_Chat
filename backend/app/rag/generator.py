from __future__ import annotations

import os

from openai import OpenAI

from app.config import OPENAI_MODEL
from app.schemas import Source
# 但是 source 最好不要只靠 LLM 生成。
# 你后端应该自己返回 sources：
# how to connect with prompts.py

def format_context(sources: list[Source]) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            f"[{index}] Dokument: {source.document}, Seite: {source.page}, "
            f"Abschnitt: {source.chunk}\n{source.text}"
        )
    return "\n\n".join(blocks)


def answer_question(question: str, sources: list[Source]) -> str:
    answer = answer_with_openai(question, sources)
    if answer:
        return answer
    return fallback_answer(sources)


def answer_with_openai(question: str, sources: list[Source]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du beantwortest Fragen ausschließlich anhand des bereitgestellten "
                    "Kontexts. Antworte auf Deutsch. Zitiere Quellen inline mit "
                    "[Dokument, Seite, Abschnitt]. Wenn die Antwort nicht im Kontext "
                    "steht, sage das klar."
                ),
            },
            {
                "role": "user",
                "content": f"Kontext:\n{format_context(sources)}\n\nFrage: {question}",
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def fallback_answer(sources: list[Source]) -> str:
    if not sources:
        return "Ich habe noch keine passenden Textstellen gefunden. Bitte lade zuerst PDFs hoch und indexiere sie."

    lines = [
        "Ohne `OPENAI_API_KEY` nutze ich eine extraktive Antwort aus den ähnlichsten Textstellen.",
        "",
        "Relevanteste Quellen:",
    ]
    for source in sources[:3]:
        excerpt = source.text[:650].strip()
        if len(source.text) > 650:
            excerpt += "..."
        lines.append(
            f"- {source.document}, Seite {source.page}, Abschnitt {source.chunk}: {excerpt}"
        )
    return "\n".join(lines)
