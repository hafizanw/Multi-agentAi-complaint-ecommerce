from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Wrapper embedding lokal, tanpa biaya API, tanpa latensi internet."""

    _instance = None  # singleton supaya model tidak di-load berkali-kali

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts, show_progress_bar=False).tolist()
