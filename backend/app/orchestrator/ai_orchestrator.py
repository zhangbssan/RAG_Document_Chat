import os
import re
from typing import Any, Dict, List

import config
from Chunks import Chunks_Embedding
from Hybrid_retrieval import HybridSearcher
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from pymilvus.model.sparse import BM25EmbeddingFunction

# ==========================================
# 核心底层：单任务 SQL 流水线 (保持德国级严谨)
# ==========================================
def clean_sql(raw_sql: str) -> str:
    cleaned = re.sub(r"```sql", "", raw_sql, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    match = re.search(r"(SELECT\s.*)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        cleaned = match.group(1)
    cleaned = cleaned.split(';')[0] + ';'
    return cleaned.strip()

def build_single_task_pipeline(db: SQLDatabase, llm: ChatOllama) -> Any:
    """处理极其单一、原子的 SQL 查询"""
    sql_prompt = PromptTemplate.from_template(
        """你是一个严谨的 SQLite 架构师。根据表结构，将这个【单一问题】转化为一句 SQLite 查询。
表结构: {schema}
规则：绝不解释，绝无 Markdown，只能输出一句 SELECT 开头的语句。
单一问题: {question}
纯净 SQL: """
    )
    
    step1_generate_sql = (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | sql_prompt
        | llm
        | StrOutputParser()
    )

    def execute_safely(sql_query: str) -> str:
        try:
            return db.run(sql_query)
        except Exception as e:
            return f"执行错误: {str(e)}"

    step2_execute = RunnableLambda(clean_sql) | RunnableLambda(execute_safely)

    # 我们把生成 SQL 和执行结果打包返回，不在这里做最终总结
    return RunnablePassthrough.assign(raw_sql=step1_generate_sql).assign(
        result=lambda inputs: step2_execute.invoke(inputs["raw_sql"])
    )

# ==========================================
# 架构师级扩展：任务拆解与大脑中枢 (Orchestrator)
# ==========================================
def break_down_query(question: str, llm: ChatOllama) -> List[str]:
    """将复杂问题拆解为多个独立的原子问题"""
    decompose_prompt = PromptTemplate.from_template(
        """你是一个任务拆解专家。用户会输入一段话，其中可能包含多个独立的查询意图。
请将它们拆解成独立的、完整的单一问题。
规则：
1. 每一行只输出一个独立的问题。
2. 不要加任何序号、破折号或多余的解释。
3. 若某个子问题需要查阅公司政策/制度/员工手册等文档（而不是业务数据库表），请在这个子问题句子开端添加“according to policy，” ，以便系统路由到知识库检索。

用户输入: {question}
拆解后的独立问题列表:"""
    )
    chain = decompose_prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"question": question})
    
    # 简单的按行清洗
    questions = [q.strip() for q in raw_output.split('\n') if q.strip() and len(q.strip()) > 2]
    
    # 过滤掉可能的 Markdown 或废话前缀
    questions = [re.sub(r"^[\d\-\*\.]+\s*", "", q) for q in questions]
    return questions

def _subquestion_needs_policy_rag(sub_q: str) -> bool:
    return "policy" in sub_q.lower()

def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=0,
        base_url=config.OLLAMA_BASE_URL,
    )

def _policy_processor_for_hybrid() -> Chunks_Embedding:
    """在本地重建与入库文本一致的 chunk，并拟合 BM25，供 sparse 查询编码。"""
    pdf_path = config.LEAVE_POLICY_FILE_PATH
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"未找到年假政策 PDF: {pdf_path}")
    processor = Chunks_Embedding()
    chunks = processor.split_text(pdf_path)
    texts = [d.page_content for d in chunks]
    bm25_ef = BM25EmbeddingFunction()
    bm25_ef.fit(texts)
    processor.bm25_ef = bm25_ef
    return processor

def _format_hybrid_snippets(combined: Dict[str, Any], max_chars: int = 6000) -> str:
    parts: List[str] = []
    for key, type_label in (("dense", "dense"), ("sparse", "sparse")):
        for i, hit in enumerate(combined.get(key, []), 1):
            content = hit.get("content") or ""
            score = hit.get("score", 0.0)
            meta = hit.get("metadata") or {}
            parts.append(
                f"[{type_label} #{i}] score={score:.4f} meta={meta}\n{content}"
            )
    text_blob = "\n\n".join(parts)
    if len(text_blob) > max_chars:
        return text_blob[:max_chars] + "\n\n... (truncated)"
    return text_blob

def _answer_from_rag_only(sub_question: str, hybrid_raw: Dict[str, Any], llm: ChatOllama) -> str:
    rag_context = _format_hybrid_snippets(hybrid_raw)
    rag_prompt = PromptTemplate.from_template(
        """你是企业政策知识库助理。你只能根据下方【检索结果摘录】作答，禁止使用预训练知识补充、猜测或推断未出现在摘录中的条款。
若摘录不足以回答，请明确说明「根据所提供的检索摘录无法完整回答」并列出摘录中实际提到的要点。

子问题: {sub_question}

检索结果摘录:
{rag_context}

仅基于摘录的回答: """
    )
    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({"sub_question": sub_question, "rag_context": rag_context})


def run_enterprise_orchestrator(question: str) -> str:
    """拆解子问题：含 policy 的走混合检索 RAG；其余走 SQL。若曾有过 RAG，则最终合并两侧材料。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db = SQLDatabase.from_uri(f"sqlite:///{os.path.join(current_dir, 'enterprise_erp.db')}")
    llm = _build_llm()

    print(f"📥 [中枢] 收到复杂用户输入: '{question}'")

    sub_questions = break_down_query(question, llm)
    print(f"🔪 [中枢] 任务拆解完毕，共 {len(sub_questions)} 个子任务:")
    for idx, sq in enumerate(sub_questions):
        route = "RAG(policy)" if _subquestion_needs_policy_rag(sq) else "SQL"
        print(f"   ├─ 子任务 {idx+1} [{route}]: {sq}")

    policy_sqs = [q for q in sub_questions if _subquestion_needs_policy_rag(q)]
    sql_sqs = [q for q in sub_questions if not _subquestion_needs_policy_rag(q)]

    sql_facts_lines: List[str] = []
    single_pipeline = build_single_task_pipeline(db, llm)
    for sq in sql_sqs:
        print(f"   ⚙️ [SQL] 正在查询: {sq}")
        res = single_pipeline.invoke({"question": sq})
        sql_facts_lines.append(f"针对问题「{sq}」，数据库查询结果是: {res['result']}")
        print(f"      ↳ 底层 SQL: {res.get('raw_sql', 'N/A').strip()}")
        print(f"      ↳ 拿到的数据: {res['result']}")

    rag_block_lines: List[str] = []
    if policy_sqs:
        print("\n📚 [RAG] 初始化政策知识库混合检索 (dense + sparse)...")
        try:
            processor = _policy_processor_for_hybrid()
            hybrid_searcher = HybridSearcher(
                processor,
                collection_name=config.LEAVE_POLICY_COLLECTION_NAME,
            )
        except Exception as e:
            err = f"政策知识库检索不可用: {e}"
            print(f"   ❌ {err}")
            for sq in policy_sqs:
                rag_block_lines.append(f"子问题「{sq}」: {err}")
        else:
            for sq in policy_sqs:
                print(f"   🔎 [RAG] Hybrid search: {sq}")
                combined = hybrid_searcher.search(sq, top_k=3)
                answer = _answer_from_rag_only(sq, combined, llm)
                rag_block_lines.append(
                    f"子问题「{sq}」\n"
                    f"— 模型归纳（仅允许基于检索摘录）: {answer}"
                )
                snippet = answer.replace("\n", " ")[:240]
                print(f"      ↳ RAG 归纳: {snippet}...")

    sql_block = (
        "\n".join(sql_facts_lines) if sql_facts_lines else "（本次无 SQL 子任务）"
    )
    rag_block = (
        "\n\n".join(rag_block_lines) if rag_block_lines else "（本次无政策 RAG 子任务）"
    )

    print("\n📝 [中枢] 正在进行最终汇总生成...")
    if policy_sqs:
        merged_prompt = PromptTemplate.from_template(
            """你是专业的企业 ERP 助理。请整合以下材料，用礼貌、专业的口吻回答用户的原始问题。
硬性要求：
1) SQL：只能使用【SQL 数据库事实】中的内容，不得编造表名或数值。
2) 政策知识库：只能使用【政策知识库(RAG)】中已给出的归纳文字；不得补充未出现在该节中的条款。
3) 若某节为空、报错或不足以回答，请在最终汇报中如实说明。

用户原始问题: {question}

【SQL 数据库事实】
{sql_block}

【政策知识库(RAG)】
{rag_block}

综合汇报: """
        )
        merged_chain = merged_prompt | llm | StrOutputParser()
        return merged_chain.invoke(
            {"question": question, "sql_block": sql_block, "rag_block": rag_block}
        )

    summarize_prompt = PromptTemplate.from_template(
        """你是一个专业的企业 ERP 助理。
请根据以下从数据库中提取的【绝对真实事实】，以礼貌、专业的口吻回答用户的原始问题。
绝不捏造事实。如果结果提示报错或空缺，请如实告知。

用户原始问题: {question}

后台数据库真实事实:
{facts}

专业汇报: """
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    return summarize_chain.invoke({"question": question, "facts": sql_block})


def main() -> None:
    complex_question = (
        "How many people are in the Engineering department in total? "
        "How many annual leave days does the employee named Baoshuang have left? "
        "According to the annual leave policy, what are the carryover rules?"
    )
    final_answer = run_enterprise_orchestrator(complex_question)

    print("\n" + "=" * 40)
    print("🛡️ 最终汇报")
    print("=" * 40)
    print(final_answer)


if __name__ == "__main__":
    main()
