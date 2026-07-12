# Enterprise Multi-Agent RAG — E-Commerce Complaint Resolution System

Sistem Multi-Agent berbasis RAG untuk resolusi komplain pelanggan E-Commerce.
Studi kasus enterprise multi-divisi: **Customer Service (Orchestrator), Finance, Logistics, dan Quality Assurance (QA)**, dengan **Evaluator Agent** independen untuk menilai kualitas jawaban akhir.

> Status: **PHASE 1 — Project Foundation** (lihat `docs/PROGRESS.md` — akan ditambahkan pada phase akhir)

## Tech Stack

| Komponen | Teknologi |
|---|---|
| Workflow Orchestration | LangGraph |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| LLM Default | meta-llama/Llama-3.2-3B-Instruct (HuggingFace) |
| API | FastAPI |
| Bahasa | Python 3.12 |

LLM dapat diganti hanya melalui konfigurasi (`.env`) tanpa mengubah kode utama, mendukung: OpenAI, Claude, Gemini, DeepSeek, Qwen, Mistral, dan Ollama.

## Struktur Project

```
app/
data/
src/
  config/
  core/
  utils/
  models/
  embeddings/
  vectordb/
  retrievers/
  prompts/
  agents/
  workflows/
  evaluation/
  api/
  services/
tests/
.env.example
.gitignore
requirements.txt
```

## Setup (akan dilengkapi di PHASE 10)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Roadmap Phase

- [x] PHASE 1 — Project Foundation
- [ ] PHASE 2 — Core Infrastructure
- [ ] PHASE 3 — Data Layer
- [ ] PHASE 4 — RAG Engine
- [ ] PHASE 5 — Domain Agents
- [ ] PHASE 6 — Orchestrator
- [ ] PHASE 7 — Evaluation
- [ ] PHASE 8 — API
- [ ] PHASE 9 — Testing
- [ ] PHASE 10 — Deployment & Documentation
