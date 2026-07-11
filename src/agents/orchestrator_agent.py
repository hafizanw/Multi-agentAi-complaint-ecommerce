import autogen
from src.agents.logistics_agent import logistics_agent
from src.agents.finance_agent import finance_agent
from src.agents.qa_agent import qa_agent
from src.config import get_llm_config

llm_config = get_llm_config()

user_proxy = autogen.UserProxyAgent(
    name="CustomerComplaint",
    human_input_mode="NEVER",
    code_execution_config=False,
    max_consecutive_auto_reply=0,
)

orchestrator = autogen.AssistantAgent(
    name="OrchestratorAgent",
    system_message=(
        "Kamu adalah orchestrator customer service pada sistem resolusi komplain "
        "e-commerce. Tugasmu:\n"
        "1. Klasifikasikan intent keluhan pelanggan.\n"
        "2. Delegasikan ke LogisticsAgent untuk urusan status pengiriman/keterlambatan.\n"
        "3. Delegasikan ke FinanceAgent untuk urusan pembayaran/refund.\n"
        "4. Delegasikan ke QAAgent untuk urusan kualitas produk/rating/review.\n"
        "5. Untuk kasus GABUNGAN (misal refund akibat keterlambatan), WAJIB minta "
        "LogisticsAgent verifikasi status pengiriman DULU sebelum FinanceAgent "
        "memutuskan kelayakan refund -- jangan biarkan FinanceAgent memutuskan sendirian.\n"
        "6. Setelah semua agent relevan memberi jawaban, rangkum jadi SATU jawaban "
        "akhir untuk pelanggan dalam Bahasa Indonesia yang sopan, sertakan alasan "
        "atau sumber data yang mendasari keputusan (explainability).\n"
        "7. Akhiri dengan kata 'SELESAI' di baris terakhir setelah rangkuman final."
    ),
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, orchestrator, logistics_agent, finance_agent, qa_agent],
    messages=[],
    max_round=10,
    speaker_selection_method="auto",
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


def handle_complaint(complaint_text: str):
    """Jalankan satu siklus resolusi komplain penuh. Return list transcript pesan."""
    groupchat.messages = []  # reset supaya tiap panggilan bersih, tidak numpuk histori lama
    user_proxy.initiate_chat(manager, message=complaint_text)
    return groupchat.messages


if __name__ == "__main__":
    import sys
    complaint = sys.argv[1] if len(sys.argv) > 1 else "Order saya telat 20 hari, saya mau refund"
    transcript = handle_complaint(complaint)
    print("\n=== TRANSCRIPT ===")
    for msg in transcript:
        print(f"[{msg.get('name', 'unknown')}]: {msg.get('content', '')[:300]}")
