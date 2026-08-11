POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


def classify_intent(query):
    query_lower = query.lower()

    for keyword in POLICY_KEYWORDS:
        if keyword in query_lower:
            return "policy_question"

    return "general_question"


def mock_llm(query, retrieved_documents):
    intent = classify_intent(query)

    if intent == "policy_question":
        if not retrieved_documents:
            return "Based on the retrieved context: No relevant policy information was found."

        return (
            "Based on the retrieved context: "
            + retrieved_documents[0]
        )

    return "I can only answer questions about Zepto policies right now."
