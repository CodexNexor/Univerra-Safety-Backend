# app/services/query.py

from app.services.classifier import classify_query
from app.services.gemini import generate_human_answer
from app.services.tavily import search_with_tavily

def process_user_query(query: str, history: list = None) -> str:
    """
    Process incoming user query and return a human-friendly answer.
    Supports two modes: general and chit-chat.

    :param query: The user's latest message
    :param history: Optional list of past messages dicts with keys: role, text, ts
    """

    # Prepare a compact history string for LLM context
    history_text = ""
    if history:
        parts = []
        for m in history:
            role = m.get("role", "user")
            text = m.get("text", "")
            parts.append(f"{role}: {text}")
        history_text = "\n".join(parts)

    # Step 1: Classify the query type
    query_type = classify_query(query)

    if query_type == "general":
        print("\n💡 Type: General Query")

        # Step 2: Search directly
        merged_result = search_with_tavily(query, is_location_query=False)

        # Step 3: Add memory context
        merged_result["history"] = history_text
        merged_result["query"] = query

        # Step 4: Humanize result
        answer = generate_human_answer(merged_result)
        return answer

    else:  # chit-chat
        print("\n💬 Type: Casual/Chit-chat")

        # Step 2: Pass query + history directly to Gemini
        return generate_human_answer({
            "query": query,
            "history": history_text
        })


# ✅ CLI Test Mode
if __name__ == "__main__":
    while True:
        user_input = input("\n🗣️ You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = process_user_query(user_input)
        print("\n🤖 UNIVERRA:", response)
