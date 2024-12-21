# Building a vector DB and making accessible for multiple users
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

user_vector_storages = {}

def create_vector_storage(user_id):
    dimension = 384;
    index = faiss.IndexFlatL2(dimension)
    user_vector_storages[user_id] = {
        "index": index,
        "text": []
    }

def store_message(user_id, message):
    embedding = model.encode([message])
    user_vector_storages[user_id]["index"].add(np.array(embedding))
    user_vector_storages[user_id]["text"].append(message)

def get_message(user_id, query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = user_vector_storages[user_id]["index"].search(query_embedding, top_k)
    return [user_vector_storages[user_id]["text"][i] for i in indices[0]]

user_id = "user_1234"
create_vector_storage(user_id)
store_message(user_id, "I have 3 apples.")
store_message(user_id, "I ate 1 apples.")
store_message(user_id, "Then I bought 4 more.")

query = "How many apples do I have?"
context = get_message(user_id, query)
print(context)
