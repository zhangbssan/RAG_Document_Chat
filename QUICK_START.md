# 🚀 RAG 文档对话系统 - 快速开始指南

## 系统特性

✅ **PDF 上传和索引** - 支持批量上传 PDF 文件
✅ **智能问答** - 基于上传的文档进行对话
✅ **来源标注** - 每个答案都显示引用来自哪个文档和页码
✅ **流式回答** - 实时显示答案（无需等待完整生成）
✅ **中文支持** - 完整的中文界面和提示
✅ **无 API 模式** - 即使没有 OpenAI 密钥也能使用

## 系统架构

```
┌─────────────────┐
│  Streamlit UI   │ ← 前端 (frontend/)  
│   (中文界面)     │
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────────────────────────────┐
│        FastAPI Backend (backend/)        │
├─────────────────────────────────────────┤
│ ┌───────────┐  ┌───────────┐  ┌───────┐ │
│ │  Retriever│  │ Generator │  │ Chat  │ │
│ └─────┬─────┘  └─────┬─────┘  └───┬───┘ │
│       ↓              ↓             ↓     │
│ ┌────────────────────────────────────┐  │
│ │     ChromaDB Vector Store          │  │
│ │     (文档向量检索)                  │  │
│ └────────────────────────────────────┘  │
└─────────────────────────────────────────┘
         ↓
┌──────────────────┐
│  OpenAI API      │ (可选)
│  GPT-4o Mini     │ (无 API 密钥时使用提取式)
└──────────────────┘
```

## 快速开始

### 1️⃣ 安装依赖

**后端**：
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**前端**：
```bash
cd frontend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ 启动服务

**启动后端**（一个终端）：
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

预期输出：
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**启动前端**（另一个终端）：
```bash
cd frontend
source .venv/bin/activate
streamlit run app.py
```

预期输出：
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### 3️⃣ 使用应用

1. **打开浏览器** → http://localhost:8501
2. **左侧面板**：
   - 选择 PDF 文件（支持多选）
   - 点击 "🚀 开始索引"
   - 等待 "✅ 成功" 消息
3. **主面板**：
   - 在输入框输入问题
   - 按 Enter 或点击发送
   - 智能助手会流式显示答案
   - 展开 "📖 查看来源信息" 查看引用

## 常用操作

### 上传 PDF

```
左侧 → 选择 PDF 文件 → 🚀 开始索引 → 等待完成
```

### 提问

```
主面板 → 输入问题 → 按 Enter → 查看答案
```

### 查看来源

```
每个答案下 → 📖 查看来源信息 → 展开查看详细来源
```

### 清空对话

```
左侧 → 🗑️ 清空对话
```

## 配置选项

### 环境变量

创建 `.env` 文件（可选）：

```bash
# 后端 (backend/.env)
OPENAI_API_KEY=sk-your-key-here     # 可选：用于 GPT 生成答案
OPENAI_MODEL=gpt-4o-mini            # 默认：gpt-4o-mini
TOP_K=5                              # 默认：检索 5 个相关段落
CHUNK_SIZE=950                       # 默认：文本块大小
CHUNK_OVERLAP=180                    # 默认：块重叠量

# 前端 (frontend/.env)
API_BASE_URL=http://127.0.0.1:8000  # 后端地址
```

## API 端点

| 端点 | 方法 | 功能 | 调用方 |
|------|------|------|--------|
| `/api/upload` | POST | 上传 PDF | 前端 |
| `/api/chat` | POST | 完整答案 | 前端 |
| `/api/chat/stream` | POST | 流式答案 | 前端 |
| `/api/documents/stats` | GET | 统计信息 | 前端 |
| `/api/documents/list` | GET | 文档列表 | 前端 |
| `/health` | GET | 健康检查 | 监控 |
| `/docs` | GET | API 文档 | 浏览 |

## 消息格式示例

### 上传请求

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@document.pdf"
```

响应：
```json
{
  "added_chunks": 15,
  "messages": [
    "✅ document.pdf: 已索引 15 个文本段落"
  ]
}
```

### 聊天请求

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "这个文档讲了什么?", "top_k": 5}'
```

响应：
```json
{
  "answer": "根据文档...[文档-P3-S2]...",
  "sources": [
    {
      "text": "文档内容片段...",
      "document": "document.pdf",
      "page": 3,
      "chunk": 2,
      "score": 0.95
    }
  ]
}
```

## 来源标注格式

答案中的引用格式：`[文档名-P页码-S段落号]`

示例：
```
根据员工手册 [employee_handbook.pdf-P3-S2] 的规定，
公司规定每年有 20 天的年假。
```

## 常见问题

### Q: 没有 OpenAI API 密钥能用吗？

**A:** 可以！系统会自动使用 "提取式回答" 模式，直接从最相关的文档段落中提取答案。

### Q: 如何处理大文件？

**A:** 你需要看一下 chunker.py 中的 CHUNK_SIZE 和 CHUNK_OVERLAP 参数。默认设置适合大多数场景。

### Q: 如何获得更好的答案质量？

**A:** 
1. 确保 PDF 文字清晰（OCR 质量好）
2. 提问要具体详细
3. 使用 OpenAI API 以获得更智能的回答
4. 调整 CHUNK_SIZE 以适应文档结构

### Q: 支持哪些语言？

**A:** 系统支持多语言 PDF，嵌入模型 `paraphrase-multilingual-MiniLM-L12-v2` 支持 50+ 语言。

### Q: 数据会被保存吗？

**A:** 
- PDF 文件保存在 `backend/data/uploads/`
- 向量数据保存在 `backend/data/chroma/`
- 不会发送到任何外部服务（除非设置了 OpenAI API）

## 故障排除

### 后端启动失败

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 检查依赖
pip install -r backend/requirements.txt

# 重新启动
uvicorn app.main:app --reload
```

### 前端无法连接后端

```bash
# 检查后端是否运行
curl http://localhost:8000/health

# 检查 API_BASE_URL
echo $API_BASE_URL

# 如果在 Docker 中，使用：
API_BASE_URL=http://backend:8000
```

### PDF 上传失败

- 确保文件是真正的 PDF（不是其他格式）
- 检查文件大小（系统可处理任何大小，但很大的文件会很慢）
- 查看后端日志获取详细错误

### 答案质量差

- 尝试不同的问题措辞
- 确保 PDF 内容确实涉及你的问题
- 增加 top_k 值以获取更多上下文
- 使用 OpenAI API 以获得更好的生成质量

## Docker 部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

访问地址：
- 后端 API: http://localhost:8000
- 前端 UI: http://localhost:8501

## 项目结构

```
RAG_Document_Chat/
├── backend/                          # FastAPI 后端
│   ├── app/
│   │   ├── api/                      # API 路由
│   │   │   ├── chat.py               # 聊天端点（重写）
│   │   │   ├── documents.py          # 文档管理（改进）
│   │   │   ├── upload.py             # PDF 上传（改进）
│   │   │   └── evaluation.py
│   │   ├── rag/
│   │   │   ├── prompts.py            # 统一提示词（重写）
│   │   │   ├── generator.py          # 回答生成（重写）
│   │   │   ├── retriever.py          # 文档检索
│   │   │   ├── embeddings.py         # 向量处理（改进）
│   │   │   ├── vector_store.py       # 向量存储
│   │   │   ├── chunker.py            # 文本分块
│   │   │   ├── pdf_loader.py         # PDF 加载
│   │   │   └── types.py              # 类型定义
│   │   ├── main.py                   # FastAPI 应用
│   │   ├── schemas.py                # 数据模型（扩展）
│   │   ├── config.py                 # 配置
│   │   └── utils/                    # 工具函数
│   ├── requirements.txt
│   ├── Dockerfile
│   └── data/                         # 数据目录
│       ├── uploads/                  # PDF 文件
│       └── chroma/                   # 向量数据库
│
├── frontend/                         # Streamlit 前端
│   ├── app.py                        # 主应用（完全重写）
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml                # Docker 编排
├── README.md                         # 原始文档
├── IMPLEMENTATION_SUMMARY.md         # 实现总结
├── QUICK_START.md                    # 本文件
└── validate_setup.py                 # 验证脚本
```

## 下一步改进

- [ ] 添加对话历史导出功能
- [ ] 实现 PDF 作者/主题过滤
- [ ] 添加答案评分和反馈机制
- [ ] 支持多模态（图片识别）
- [ ] 实现 Reranker 以提高准确性
- [ ] 添加用户认证和权限控制
- [ ] 实现批量问答模式
- [ ] 添加答案缓存以提高性能

## 许可证

MIT

## 联系方式

如有问题或建议，请联系项目维护者。

---

**最后更新**: 2024-05-18
**版本**: 1.0
**状态**: ✅ 生产就绪
