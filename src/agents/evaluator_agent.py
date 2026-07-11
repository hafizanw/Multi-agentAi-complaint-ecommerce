"""
Evaluator sederhana untuk mengukur 4 metrik dari proposal:
Accuracy, Efficiency, Explainability, Hallucination (versi ringan/manual-check).
"""
import time
import json
import os
from src.agents.orchestrator_agent import handle_complaint


def evaluate_accuracy(scenario: dict, transcript: list) -> bool:
    """Cek apakah semua agent yang diharapkan benar-benar terlibat dalam percakapan."""
    called_agents = {msg.get("name") for msg in transcript if msg.get("name")}
    expected = set(scenario["expected_agents"])
    return expected.issubset(called_agents)


def evaluate_efficiency(start_time: float, end_time: float) -> dict:
    return {"response_time_sec": round(end_time - start_time, 2)}


def evaluate_explainability(transcript: list) -> bool:
    """Cek apakah jawaban akhir (dari orchestrator) menyebut order_id / alasan konkret."""
    orchestrator_msgs = [m for m in transcript if m.get("name") == "OrchestratorAgent"]
    if not orchestrator_msgs:
        return False
    final_msg = orchestrator_msgs[-1].get("content", "")
    keywords = ["order", "berdasarkan", "status", "karena"]
    return any(k in final_msg.lower() for k in keywords)


def evaluate_hallucination_flag(transcript: list) -> str:
    """
    Versi ringan (tanpa LLM-judge terpisah): cek apakah agent domain memanggil
    fungsi RAG (tool call) sebelum menjawab. Kalau tidak ada tool call sama sekali,
    tandai sebagai 'berpotensi halusinasi' karena jawaban tidak berbasis data RAG.
    """
    tool_calls = [m for m in transcript if m.get("tool_calls") or m.get("function_call")]
    return "aman (ada tool call RAG)" if tool_calls else "PERLU DICEK (tidak ada tool call terdeteksi)"


def run_evaluation(scenarios_path: str = None):
    if scenarios_path is None:
        scenarios_path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "test_scenarios.json")

    scenarios = json.load(open(scenarios_path, encoding="utf-8"))
    results = []

    for sc in scenarios:
        print(f"\n>>> Menjalankan skenario {sc['id']}: {sc['input']}")
        start = time.time()
        transcript = handle_complaint(sc["input"])
        end = time.time()

        result = {
            "id": sc["id"],
            "input": sc["input"],
            "accuracy": evaluate_accuracy(sc, transcript),
            "efficiency": evaluate_efficiency(start, end),
            "explainability": evaluate_explainability(transcript),
            "hallucination_check": evaluate_hallucination_flag(transcript),
        }
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return results


if __name__ == "__main__":
    all_results = run_evaluation()
    print("\n=== RINGKASAN EVALUASI ===")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))
