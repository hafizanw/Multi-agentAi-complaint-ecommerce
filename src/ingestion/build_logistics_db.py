"""
Build ChromaDB vectorstore untuk Logistics Agent.
Sumber: olist_orders_dataset.csv + olist_order_items_dataset.csv (+ products untuk konteks nama produk)
"""
import pandas as pd
import chromadb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag.embedder import LocalEmbedder
from src.config import RAW_DIR, VECTORSTORE_PATHS, COLLECTION_NAMES, SAMPLE_SIZE


def load_and_merge():
    orders = pd.read_csv(f"{RAW_DIR}/olist_orders_dataset.csv", encoding="utf-8-sig")
    items = pd.read_csv(f"{RAW_DIR}/olist_order_items_dataset.csv", encoding="utf-8-sig")
    products = pd.read_csv(f"{RAW_DIR}/olist_products_dataset.csv", encoding="utf-8-sig")
    translation = pd.read_csv(f"{RAW_DIR}/product_category_name_translation.csv", encoding="utf-8-sig")

    # sampling di level order_id supaya representatif dan cepat untuk development
    if SAMPLE_SIZE:
        order_ids_sample = orders["order_id"].drop_duplicates().sample(
            n=min(SAMPLE_SIZE, orders["order_id"].nunique()), random_state=42
        )
        orders = orders[orders["order_id"].isin(order_ids_sample)]

    products = products.merge(translation, on="product_category_name", how="left")
    merged = orders.merge(items, on="order_id", how="left")
    merged = merged.merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
    return merged


def build_documents(merged: pd.DataFrame):
    docs, ids, metadatas = [], [], []

    for order_id, group in merged.groupby("order_id"):
        row = group.iloc[0]
        n_items = len(group)
        categories = group["product_category_name_english"].dropna().unique().tolist()
        categories_str = ", ".join(categories) if categories else "tidak diketahui"

        delivered = row.get("order_delivered_customer_date")
        delivered_str = delivered if pd.notna(delivered) else "belum diterima pelanggan"

        text = (
            f"Order ID {order_id} berstatus '{row['order_status']}'. "
            f"Dibeli pada {row['order_purchase_timestamp']}. "
            f"Estimasi tanggal tiba: {row['order_estimated_delivery_date']}. "
            f"Tanggal diterima aktual: {delivered_str}. "
            f"Dikirim ke kurir pada: {row.get('order_delivered_carrier_date', 'belum dikirim')}. "
            f"Jumlah item dalam order: {n_items}. "
            f"Kategori produk: {categories_str}."
        )

        docs.append(text)
        ids.append(str(order_id))
        metadatas.append({
            "order_status": str(row["order_status"]),
            "estimated_delivery": str(row["order_estimated_delivery_date"]),
            "delivered_date": str(delivered_str),
        })

    return docs, ids, metadatas


def build_logistics_vectorstore():
    print("Loading & merging data logistik...")
    merged = load_and_merge()

    print("Menyusun dokumen naratif per order...")
    docs, ids, metadatas = build_documents(merged)
    print(f"Total dokumen: {len(docs)}")

    print("Membuat embedding (all-MiniLM-L6-v2, lokal)...")
    embedder = LocalEmbedder()
    embeddings = embedder.embed(docs)

    print("Menyimpan ke ChromaDB...")
    client = chromadb.PersistentClient(path=VECTORSTORE_PATHS["logistics"])
    # hapus collection lama jika ada, supaya bisa re-run tanpa duplikat
    try:
        client.delete_collection(COLLECTION_NAMES["logistics"])
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAMES["logistics"])

    batch_size = 4000
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Logistics vectorstore selesai: {collection.count()} dokumen ter-index.")


if __name__ == "__main__":
    build_logistics_vectorstore()
