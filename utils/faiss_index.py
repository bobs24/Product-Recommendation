import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.image_map = []

    def build(self, embeddings, image_ids):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-10)
        self.index.add(normalized_embeddings.astype('float32'))
        self.image_map = image_ids

    def search(self, query_vector, threshold=0.8, k=50):
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        query = np.array([query_vector]).astype('float32')
        similarities, indices = self.index.search(query, k)

        results = []
        for i, sim in zip(indices[0], similarities[0]):
            if sim >= threshold:
                results.append((self.image_map[i], sim))
        return sorted(results, key=lambda x: -x[1])