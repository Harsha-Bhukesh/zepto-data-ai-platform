# Module 3 — Zepto Support Assistant

## Overview

The **Zepto Support Assistant** is a small Retrieval-Augmented Generation (RAG) service that answers customer questions using a fixed corpus of Zepto policy documents.

The system demonstrates:

* Document ingestion
* Local text embeddings
* ChromaDB vector storage
* Semantic retrieval
* Intent classification
* LangGraph workflow orchestration
* Structured prompting
* Deterministic offline mock LLM behavior
* Pydantic response validation
* FastAPI API serving
* Docker containerization

The required graded implementation works completely offline using the `MOCK_LLM` baseline. No external LLM API key or account is required.

---

# 1. Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── ingest.py
├── retrieval.py
├── prompt.py
├── mock_llm.py
├── schema.py
├── graph.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

# 2. Policy Document Corpus

The application uses eight Zepto policy documents.

| File         | Policy                    |
| ------------ | ------------------------- |
| `doc_01.txt` | Delivery Policy           |
| `doc_02.txt` | Returns & Refunds         |
| `doc_03.txt` | Membership Tiers          |
| `doc_04.txt` | Order Tracking            |
| `doc_05.txt` | Order Cancellation Policy |
| `doc_06.txt` | Damaged or Missing Items  |
| `doc_07.txt` | Gift Cards                |
| `doc_08.txt` | Customer Support Hours    |

Each document is used as a retrieval source for answering policy-related customer questions.

---

# 3. System Architecture

The complete system follows this flow:

```text
                         User Query
                             │
                             ▼
                       FastAPI /ask
                             │
                             ▼
                    LangGraph StateGraph
                             │
                             ▼
                     classify_intent
                        /          \
                       /            \
                      ▼              ▼
             policy_question    general_question
                    │                  │
                    ▼                  ▼
          retrieve_and_answer     direct_answer
                    │                  │
                    ▼                  │
                ChromaDB              │
                    │                  │
                    ▼                  │
                 Top-3                 │
               Retrieval              │
                    │                  │
                    ▼                  │
              Mock / Real LLM         │
                    │                  │
                    └────────┬─────────┘
                             ▼
                    Pydantic Validation
                             │
                             ▼
                       JSON Response
```

---

# 4. RAG Pipeline

The RAG pipeline consists of four major stages:

```text
Ingestion
    │
    ▼
Embedding
    │
    ▼
Retrieval
    │
    ▼
Generation
```

## Stage 1 — Ingestion

The files in `docs/` are loaded by `ingest.py`.

Each policy document is read and converted into a retrieval unit.

Because the documents are short, a simple per-document chunking strategy is used.

---

## Stage 2 — Embedding

The project uses the local Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The model converts each document chunk into a numerical vector.

No external embedding API is required.

The vectors are stored in a ChromaDB collection named:

```text
zepto_policies
```

---

## Stage 3 — Retrieval

For a policy-related query, the `retrieve_and_answer` LangGraph node sends the query to the retrieval function.

The query is embedded using the same:

```text
all-MiniLM-L6-v2
```

model.

ChromaDB then performs semantic similarity search and returns the **top 3 most similar chunks**.

The retrieved document/chunk IDs are preserved and returned through the `sources` field.

---

## Stage 4 — Generation

For a `policy_question`, the retrieved context is passed to the answer-generation stage.

When `MOCK_LLM` is at its default value, no external LLM is called.

Instead, the deterministic mock response follows:

```text
Based on the retrieved context: <top retrieved chunk excerpt>
```

For a `general_question`, the system returns the fixed mock response:

```text
I can only answer questions about Zepto policies right now.
```

---

# 5. MOCK_LLM

The required graded baseline uses deterministic mock behavior.

The environment variable is:

```text
MOCK_LLM
```

The default behavior is:

```text
MOCK_LLM unset
```

or:

```text
MOCK_LLM=1
```

Both mean that the application uses the offline mock implementation.

No external LLM API call is made.

## Optional Real LLM Mode

A real LLM implementation can optionally be enabled with:

```text
MOCK_LLM=0
```

This path is optional and is not required for the graded baseline.

The required submission must work correctly with `MOCK_LLM` unset or set to `1`.

---

# 6. Intent Classification

The `classify_intent` LangGraph node determines whether retrieval is required.

In mock mode, classification is performed using a deterministic keyword heuristic.

The following keywords identify a policy question:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

If the lowercased query contains one of these keywords, the query is classified as:

```text
policy_question
```

Otherwise it is classified as:

```text
general_question
```

## Example — Policy Question

```text
What is the delivery fee?
```

Classification:

```text
policy_question
```

The graph routes the query to:

```text
retrieve_and_answer
```

## Example — General Question

```text
What is Python?
```

Classification:

```text
general_question
```

The graph routes the query to:

```text
direct_answer
```

The routing itself does not depend on `MOCK_LLM`.

---

# 7. LangGraph Workflow

The application uses LangGraph's `StateGraph`.

The graph contains the three required nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The workflow is:

```text
START
  │
  ▼
classify_intent
  │
  ├───────────────┐
  │               │
  ▼               ▼
policy_question   general_question
  │               │
  ▼               ▼
retrieve_and_     direct_answer
answer
  │               │
  └───────┬───────┘
          │
          ▼
         END
```

## `classify_intent`

Determines whether the incoming query requires retrieval.

## `retrieve_and_answer`

For policy questions:

1. Embed the query.
2. Search ChromaDB.
3. Retrieve the top 3 chunks.
4. Store the retrieved IDs.
5. Generate the answer.
6. Validate the final response.

## `direct_answer`

For general questions:

1. Skip retrieval.
2. Return the fixed mock response.
3. Set `sources` to an empty list.
4. Validate the final response.

---

# 8. Structured Prompt

The project contains a structured prompt template for the optional real-LLM path.

The prompt follows the required:

```text
Role
Context
Task
Format
Length
```

structure.

It also contains:

### Negative Constraint

The model must not answer using information that is not present in the supplied context.

For example:

```text
Do not use information that is not contained in the provided context.
```

### Few-Shot Example

The prompt includes an example demonstrating how a Zepto policy question should be answered using retrieved context.

This prompt is used by the optional `MOCK_LLM=0` generation path.

The required mock path does not make an LLM call.

---

# 9. ChromaDB

ChromaDB is used as the local vector database.

The collection used by the application is:

```text
zepto_policies
```

The documents are embedded using:

```text
all-MiniLM-L6-v2
```

and stored in the ChromaDB collection.

For policy queries, the retrieval stage returns the top three most similar chunks.

---

# 10. Pydantic Response Schema

The final API response is validated using a Pydantic model.

The response contains three fields:

```text
answer
sources
confidence
```

Example:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials...",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

## Fields

### `answer`

The final answer returned to the customer.

Type:

```text
string
```

### `sources`

The IDs of the retrieved chunks/documents used to produce the answer.

Type:

```text
list[string]
```

For general questions:

```json
"sources": []
```

### `confidence`

A value between:

```text
0.0
```

and:

```text
1.0
```

In mock mode, the implementation uses a deterministic confidence value.

---

# 11. FastAPI

The FastAPI application exposes:

```text
POST /ask
```

## Request

Example request:

```json
{
  "query": "What is the delivery fee?"
}
```

## Response

Example response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

---

# 12. Installation

Create and activate a Python environment if desired.

Then install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
fastapi
uvicorn
pydantic
chromadb
sentence-transformers
langgraph
langchain-core
```

---

# 13. Build the Vector Database

Before running the API, ingest the policy documents.

Run:

```bash
python ingest.py
```

This process:

```text
docs/*.txt
     │
     ▼
Load documents
     │
     ▼
Create chunks
     │
     ▼
Generate embeddings
     │
     ▼
Store in ChromaDB
```

The eight policy documents should be available in the ChromaDB collection after ingestion.

---

# 14. Run the FastAPI Application

Run the application locally using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# 15. Example API Call — Policy Question

With `MOCK_LLM` left at its default:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d "{\"query\":\"What is the delivery fee?\"}"
```

Example response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation...",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

The exact answer/source ordering may depend on the retrieval results produced by the local ChromaDB instance.

---

# 16. Example API Call — General Question

Request:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d "{\"query\":\"What is Python?\"}"
```

Expected mock response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This query is classified as:

```text
general_question
```

and therefore does not perform retrieval.

---

# 17. Testing the Intent Router

## Policy Query

Input:

```text
What is the delivery fee?
```

Expected classification:

```text
policy_question
```

Expected route:

```text
classify_intent
        │
        ▼
retrieve_and_answer
```

## General Query

Input:

```text
What is Python?
```

Expected classification:

```text
general_question
```

Expected route:

```text
classify_intent
        │
        ▼
direct_answer
```

---

# 18. Docker

The project includes a `Dockerfile` for running the FastAPI application in a container.

## Build

From the `support_assistant` directory:

```bash
docker build -t zepto-support-assistant .
```

## Run

```bash
docker run -p 8000:8000 zepto-support-assistant
```

The API should then be accessible at:

```text
http://127.0.0.1:8000
```

The `/ask` endpoint can be tested using the same request described above.

The required Docker implementation is intended to work using the offline mock baseline.

---

# 19. Data Flow

The complete data flow is:

```text
                    ┌──────────────────┐
                    │  Policy Documents│
                    │     docs/*.txt   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   ingest.py      │
                    │ Load + Chunk     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Sentence         │
                    │ Transformers     │
                    │ all-MiniLM-L6-v2 │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │ zepto_policies   │
                    └────────┬─────────┘
                             │
                             │ Query
                             ▼
                    ┌──────────────────┐
                    │ classify_intent  │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
             policy_question   general_question
                    │                 │
                    ▼                 ▼
          retrieve_and_answer    direct_answer
                    │                 │
                    ▼                 │
              Top-3 chunks            │
                    │                 │
                    ▼                 │
             Mock / Real LLM          │
                    │                 │
                    └────────┬────────┘
                             ▼
                    ┌──────────────────┐
                    │ Pydantic Schema  │
                    │ answer/sources/  │
                    │ confidence       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ FastAPI Response │
                    └──────────────────┘
```

---

# 20. MOCK_LLM Behavior

The `MOCK_LLM` toggle affects the **generation/classification behavior**, while retrieval continues to use the real local embedding and ChromaDB pipeline.

## Default Mode

```text
MOCK_LLM unset
```

or:

```text
MOCK_LLM=1
```

Behavior:

```text
Intent classification → keyword heuristic
Retrieval             → real ChromaDB retrieval
Answer generation     → deterministic mock response
```

No external LLM API is called.

## Optional Real-LLM Mode

```text
MOCK_LLM=0
```

Behavior:

```text
Intent classification → LLM
Retrieval             → real ChromaDB retrieval
Answer generation     → LLM using structured prompt
```

If the real LLM output does not satisfy the required Pydantic schema, the implementation retries with a corrective instruction up to two additional times before returning an error response.

The real-LLM path is optional and is not required for the graded baseline.

---

# 21. Why RAG Is Used

A fixed policy corpus is used instead of allowing the assistant to answer from general knowledge.

The RAG architecture helps ensure that policy-related responses are grounded in the provided Zepto documents.

For a policy query:

```text
User Query
    ↓
Intent Classification
    ↓
Retrieve Relevant Policy
    ↓
Use Retrieved Context
    ↓
Generate Answer
    ↓
Return Sources
```

This allows the response to be associated with the policy documents used during retrieval.

---

# 22. Module 3 Acceptance Checklist

The following requirements are addressed by this module:

* [x] Eight Zepto policy documents
* [x] Local `all-MiniLM-L6-v2` embeddings
* [x] ChromaDB vector collection
* [x] Top-3 semantic retrieval
* [x] Structured prompt
* [x] Role–Context–Task–Format–Length structure
* [x] Negative constraint
* [x] Few-shot example
* [x] LangGraph `StateGraph`
* [x] `TypedDict` state
* [x] `classify_intent` node
* [x] `retrieve_and_answer` node
* [x] `direct_answer` node
* [x] Conditional routing
* [x] Deterministic `MOCK_LLM` baseline
* [x] Pydantic structured response
* [x] `answer` field
* [x] `sources` field
* [x] `confidence` field
* [x] FastAPI `POST /ask`
* [x] Local Uvicorn execution
* [x] Dockerfile
* [x] RAG architecture documentation
* [x] Mock and optional real-LLM behavior documented

---

# 23. Optional Extensions

The following are optional and are not required for the graded baseline:

### Real LLM

Set:

```bash
MOCK_LLM=0
```

and configure an appropriate LLM provider/API key through environment variables.

API keys should never be hardcoded into source code or committed to GitHub.

### Hugging Face Spaces

The Dockerized application can optionally be deployed to Hugging Face Spaces using the free community CPU tier.

This deployment is optional and is not required for full marks.

---

# 24. Security Notes

Do not commit API keys or secrets to GitHub.

For example, do not place:

```text
GROQ_API_KEY=...
```

directly inside Python source code.

Use environment variables or platform secrets instead.

The required `MOCK_LLM` baseline does not require an LLM API key.

---

# 25. Conclusion

Module 3 implements a complete offline-capable RAG support assistant for Zepto.

The final architecture is:

```text
8 Policy Documents
        ↓
     Ingestion
        ↓
Local Embeddings
        ↓
     ChromaDB
        ↓
 Intent Classification
        ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Policy         General
Question       Question
 │               │
 ▼               ▼
Retrieve       Direct
Top-3          Answer
 │
 ▼
Mock / Real Generation
 │
 ▼
Pydantic Validation
 │
 ▼
FastAPI /ask
 │
 ▼
Structured JSON
```

The required baseline runs with `MOCK_LLM` unset or set to `1`, requiring no external LLM API key or network access to an LLM provider.
