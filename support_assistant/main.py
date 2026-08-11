from fastapi import FastAPI
from pydantic import BaseModel

from graph import app
from schema import SupportResponse


api = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str


@api.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest):
    result = app.invoke({
        "query": request.query,
        "intent": "",
        "retrieved_documents": [],
        "sources": [],
        "answer": "",
        "confidence": 0.0
    })

    response = SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

    return response
