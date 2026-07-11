"""
Build ChromaDB vectorstore untuk Finance Agent.
Sumber: olist_order_payments_dataset.csv (+ order_status dari orders untuk konteks refund)
"""
import pandas as pd
import chromadb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag.embedder import LocalEmbedder
from src.config import RAW_DIR, VECTORSTORE_PATHS, COLLECTION_NAMES, SAMPLE_SIZE


def load_and_merge():
    payments = pd.read_csv(f"{RAW_DIR}/olist_order_payments_dataset.csv", encoding="utf-8-sig")
    orders = pd.read_csv(f"{RAW_DIR}/olist_orders_dataset.csv", encoding="utf-8-sig")

    if SAMPLE_SIZE:
        order_ids_sample = orders["order_id"].drop_duplicates().sample(
            n=min(SAMPLE_SIZE, orders["order_id"].nunique()), random_state=42
        )
        payments = payments[payments["order_id"].isin(order_ids_sample)]
        orders = orders[orders["order_id"].isin(order_ids_sample)]

    merged = payments.merge(
        orders[["order_id", "order_status", "order_purchase_timestamp"]],
        on="order_id", how="left"
    )
    return merged


def build_documents(merged: pd.DataFrame):
    docs, ids, metadatas = [], [], []

    for order_id, group in merged.groupby("order_id"):
        total_paid = group["payment_value"].sum()
        methods = group["payment_type"].unique().tolist()
        max_installments = group["payment_installments"].max()
        status = group["order_status"].iloc[0]

        text = (
            f"Order ID {order_id} dibayar total {total_paid:.2f} menggunakan metode {', '.join(methods)}. "
            f"Cicilan maksimum: {max_installments}x. "
            f"Status order saat ini: '{status}'. "
            f"Jumlah transaksi pembayaran tercatat: {len(group)}."
        )

        docs.append(text)
        ids.append(str(order_id))
        metadatas.append({
            "total_paid": float(total_paid),
            "payment_methods": ", ".join(methods),
            "order_status": str(status),
        })

    return docs, ids, metadatas


def build_finance_vectorstore():
    print("Loading & merging data finance...")
    merged = load_and_merge()

    print("Menyusun dokumen naratif per order...")
    docs, ids, metadatas = build_documents(merged)
    print(f"Total dokumen: {len(docs)}")

    print("Membuat embedding (all-MiniLM-L6-v2, lokal)...")
    embedder = LocalEmbedder()
    embeddings = embedder.embed(docs)

    print("Menyimpan ke ChromaDB...")
    client = chromadb.PersistentClient(path=VECTORSTORE_PATHS["finance"])
    try:
        client.delete_collection(COLLECTION_NAMES["finance"])
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAMES["finance"])

    batch_size = 4000
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Finance vectorstore selesai: {collection.count()} dokumen ter-index.")


if __name__ == "__main__":
    build_finance_vectorstore()
