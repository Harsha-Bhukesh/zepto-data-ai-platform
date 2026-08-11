from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END

from retrieval import retrieve_documents
from mock_llm import classify_intent, mock_llm


class SupportState(TypedDict):
    query: str
    intent: str
    retrieved_documents: List[str]
    sources: List[str]
    answer: str
    confidence: float


def classify_intent_node(state: SupportState):
    intent = classify_intent(state["query"])

    return {
        "intent": intent
    }


def retrieve_and_answer_node(state: SupportState):
    results = retrieve_documents(
        state["query"],
        n_results=3
    )

    documents = results["documents"][0]
    sources = results["ids"][0]

    answer = mock_llm(
        state["query"],
        documents
    )

    return {
        "retrieved_documents": documents,
        "sources": sources,
        "answer": answer,
        "confidence": 1.0
    }


def direct_answer_node(state: SupportState):
    return {
        "retrieved_documents": [],
        "sources": [],
        "answer": "I can only answer questions about Zepto policies right now.",
        "confidence": 1.0
    }


def route_query(state: SupportState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


graph = StateGraph(SupportState)

graph.add_node(
    "classify_intent",
    classify_intent_node
)

graph.add_node(
    "retrieve_and_answer",
    retrieve_and_answer_node
)

graph.add_node(
    "direct_answer",
    direct_answer_node
)

graph.add_edge(
    START,
    "classify_intent"
)

graph.add_conditional_edges(
    "classify_intent",
    route_query,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph.add_edge(
    "retrieve_and_answer",
    END
)

graph.add_edge(
    "direct_answer",
    END
)

app = graph.compile()
