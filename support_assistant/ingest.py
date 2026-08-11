import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="zepto_policies"
)

documents = []
ids = []

for filename in sorted(os.listdir(DOCS_DIR)):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read().strip()

        documents.append(text)
        ids.append(filename.replace(".txt", ""))

print("Documents found:", len(documents))
print("Document IDs:", ids)

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents indexed successfully!")
print("Number of documents:", collection.count())
