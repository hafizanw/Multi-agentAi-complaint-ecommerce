import autogen
from src.rag.retriever import Retriever
from src.config import get_llm_config

_retriever = Retriever(domain="qa")


def search_qa(query: str) -> str:
    """Cari riwayat ulasan/rating produk berdasarkan pertanyaan pelanggan tentang order."""
    docs, metas = _retriever.query(query, top_k=3)
    if not docs:
        return "Tidak ditemukan data ulasan yang relevan untuk pertanyaan ini."
    return "\n".join(f"- {d}" for d in docs)


def build_qa_agent():
    llm_config = get_llm_config()

    agent = autogen.AssistantAgent(
        name="QAAgent",
        system_message=(
            "Kamu adalah agent Quality Assurance pada sistem resolusi komplain "
            "e-commerce. Tugasmu memberikan konteks riwayat rating dan ulasan "
            "produk/order, serta mendeteksi pola keluhan kualitas berulang. "
            "SELALU panggil fungsi search_qa untuk mengambil data faktual "
            "sebelum menjawab. Sertakan order_id dan skor rating sebagai bukti "
            "jawaban. Jika kamu melihat pola review_score rendah berulang untuk "
            "produk/order sejenis, sebutkan itu sebagai insight untuk orchestrator."
        ),
        llm_config=llm_config,
    )
    agent.register_for_llm(
        name="search_qa",
        description="Cari riwayat ulasan dan rating sebuah order berdasarkan pertanyaan pelanggan",
    )(search_qa)
    return agent


qa_agent = build_qa_agent()
