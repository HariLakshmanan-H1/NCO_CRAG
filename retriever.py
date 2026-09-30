import faiss  # Facebook AI Similarity Search library for vector indexing
import pickle  # serialize/deserialize Python objects to disk
import numpy as np  # array operations for embedding vectors
from sentence_transformers import SentenceTransformer  # pre-trained embedding model
from pathlib import Path  # file path utilities

BASE_DIR = Path(__file__).resolve().parent  # project root for this module
EMB_DIR = BASE_DIR / "Data" / "embeddings"  # directory containing saved FAISS index and metadata

class NCORetriever:

    def __init__(self):

        # Load the pre-built FAISS similarity index from disk
        self.index = faiss.read_index(str(EMB_DIR / "nco2015.faiss"))

        # Load the corresponding metadata records for each vector
        with open(EMB_DIR / "nco2015_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        # Load the same sentence embedding model used during ingestion
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loaded vectors:", self.index.ntotal)
        print("Loaded metadata:", len(self.metadata))

    def search(self, query: str, top_k: int = 5):
        """Search the FAISS index for the most similar NCO roles to the query."""

        # Embed the user query using the same model as the index
        q_emb = self.model.encode([query]).astype("float32")

        # Perform nearest-neighbor search in the FAISS index
        distances, indices = self.index.search(q_emb, top_k)

        results = []

        for idx, dist in zip(indices[0], distances[0]):
            job = self.metadata[idx]  # retrieve the original role metadata

            results.append({
                "nco_2015": job["nco_2015"],
                "title": job["title"],
                "description": job.get("description", "")[:500],
                "score": float(dist)  # FAISS distance score
            })

        return results