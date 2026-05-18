#!/usr/bin/env python3
"""
Quick validation script to check if all imports work correctly.
Run this to verify the setup before running the full application.
"""
import sys
from pathlib import Path
_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

def test_imports():
    """Test critical imports."""
    print("🔍 检查导入...")
    
    try:
        print("  ✓ 导入 prompts...")
        from app.rag import prompts
        assert hasattr(prompts, 'SYSTEM_PROMPT')
        assert hasattr(prompts, 'ANSWER_GENERATION_PROMPT')
        
        print("  ✓ 导入 generator...")
        from app.rag import generator
        assert hasattr(generator, 'answer_question')
        assert hasattr(generator, 'answer_question_stream')
        
        print("  ✓ 导入 chat API...")
        from app.api import chat
        assert hasattr(chat, 'router')
        
        print("  ✓ 导入 upload API...")
        from app.api import upload
        assert hasattr(upload, 'router')

        print("  ✓ 导入 vector store...")
        from app.rag import vector_store
        assert hasattr(vector_store, 'get_collection')
        assert hasattr(vector_store, 'indexed_file_hashes')
        assert hasattr(vector_store, 'add_chunks')
        
        print("  ✓ 导入 schemas...")
        from app.schemas import ChatRequest, ChatResponse, DocumentStats
        
        print("  ✓ 导入 retriever...")
        from app.rag import retriever
        assert hasattr(retriever, 'search_sources')
        
        print("\n✅ 所有导入检查通过！\n")
        return True
    
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}\n")
        return False
    except AssertionError as e:
        print(f"\n❌ 断言失败: {e}\n")
        return False


def test_prompts():
    """Test prompts content."""
    print("📝 检查 Prompts...")
    
    try:
        from app.rag import prompts
        
        # Check key prompts exist
        assert prompts.SYSTEM_PROMPT
        assert prompts.ANSWER_GENERATION_PROMPT
        assert prompts.NO_SOURCES_FOUND_MESSAGE
        
        # Check templates
        assert "{question}" in prompts.ANSWER_GENERATION_PROMPT
        assert "{context}" in prompts.ANSWER_GENERATION_PROMPT
        
        print("  ✓ 系统提示词")
        print("  ✓ 答案生成模板")
        print("  ✓ 无来源消息")
        print("  ✓ 模板变量")
        
        print("\n✅ Prompts 检查通过！\n")
        return True
    
    except (AssertionError, AttributeError) as e:
        print(f"\n❌ Prompts 检查失败: {e}\n")
        return False


def test_api_routes():
    """Test API route definitions."""
    print("🛣️ 检查 API 路由...")
    
    try:
        from app.api import chat, upload
        
        # Check routes exist
        routes = [route.path for route in chat.router.routes]
        assert "/chat" in routes or any("chat" in r for r in routes)
        
        print("  ✓ Chat 路由")
        upload_routes = [route.path for route in upload.router.routes]
        assert "/upload" in upload_routes or any("upload" in r for r in upload_routes)

        print("  ✓ Upload 路由")
        
        print("\n✅ API 路由检查通过！\n")
        return True
    
    except ImportError as e:
        print(f"\n❌ API 路由导入失败: {e}\n")
        return False
    except (AssertionError, AttributeError) as e:
        print(f"\n❌ API 路由检查失败: {e}\n")
        return False


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 RAG 系统验证脚本")
    print("="*50 + "\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_prompts()
    all_passed &= test_api_routes()
    
    print("="*50)
    if all_passed:
        print("✅ 所有检查通过！系统已准备好运行")
    else:
        print("❌ 某些检查失败，请检查错误信息")
    print("="*50 + "\n")
