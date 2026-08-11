import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="zepto_policies"
)


def retrieve_documents(query, n_results=3):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return results


if __name__ == "__main__":
    query = "What is the delivery fee?"

    results = retrieve_documents(query)

    print("Query:", query)
    print("\nRetrieved document IDs:")
    print(results["ids"])

    print("\nRetrieved documents:")

    for i, document in enumerate(results["documents"][0], start=1):
        print(f"\n--- Result {i} ---")
        print(document)
