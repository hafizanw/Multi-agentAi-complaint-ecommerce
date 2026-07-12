# Enterprise Multi-Agent RAG — E-Commerce Complaint Resolution System

Sistem Multi-Agent berbasis RAG untuk resolusi komplain pelanggan E-Commerce.
Studi kasus enterprise multi-divisi: **Customer Service (Orchestrator), Finance, Logistics, dan Quality Assurance (QA)**, dengan **Evaluator Agent** independen untuk menilai kualitas jawaban akhir.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # lalu isi sesuai provider LLM pilihan (ollama/openai/anthropic)
```

## Taruh Dataset

Salin 9 file CSV Olist ke folder `data/raw/`.

## Build Vectorstore (jalan sekali di awal, atau tiap dataset berubah)

```bash
python -m src.ingestion.build_logistics_db
python -m src.ingestion.build_finance_db
python -m src.ingestion.build_qa_db
```

## Jalankan Sistem

```bash
# Uji satu agent
python -c "from src.agents.logistics_agent import search_logistics; print(search_logistics('status pengiriman'))"

# Uji orchestrator penuh
python -m src.agents.orchestrator_agent "Order saya telat 20 hari, saya mau refund"

# Evaluasi otomatis
python -m src.agents.evaluator_agent

# API
uvicorn app.api:app --reload --port 8000

# Demo visual
streamlit run app/streamlit_demo.py
```

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