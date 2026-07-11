# Multi-Agent Complaint Resolution System (RAG + AutoGen)

Sistem resolusi komplain e-commerce berbasis multi-agent yang berkolaborasi,
menggunakan RAG (ChromaDB + all-MiniLM-L6-v2) tanpa fine-tuning.

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

## Struktur

Lihat dokumentasi arsitektur di proposal project (`Sistem_Multi-Agent_Komplain_E-Commerce.pdf`).
