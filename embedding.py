# Building a vector DB and searching in it.
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    "I have 3 apples.",
    "I ate 1 apple.",
    "Then I bought 4 more apples."
]

embeddings = model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) 

val = np.array(embeddings)
index.add(val)

query1 = "I ate 2 apple."
query_embedding1 = model.encode([query1])
index.add(query_embedding1)


query2 = "How many apples do I have?"
query_embedding = model.encode([query2])

k = 2
distances, indices = index.search(query_embedding, k)

similar_texts = [texts[i] for i in indices[0]]
print("Similar texts:", similar_texts)
