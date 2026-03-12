# app/services/classifier.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# UNIVERRA - Nexor's Robust Classifier
def classify_query(user_query: str) -> str:
    """
    Classifies a user query into:
    - 'general'  → fact-based questions, news, weather, general info, search topics
    - 'chitchat' → casual, personal, or social conversation
    """

    prompt = f"""
You are UNIVERRA, a professional AI created by Nexor. 
Your task is to classify the following user query into **one of two categories only**:

1. general  → fact-based questions, knowledge, news, weather, or topic-specific information
2. chitchat → greetings, jokes, casual talk, questions about you or your creator

💡 Examples:
- "What is a tsunami?" → general  
- "What happened in Ukraine?" → general  
- "Tell me a joke" → chitchat  
- "Hi, who created you?" → chitchat  
- "Are you a robot?" → chitchat  

Now classify this query using only one word (general or chitchat):

Query: "{user_query}"
"""

    try:
        response = model.generate_content(prompt)
        category = response.text.strip().lower()

        if "chitchat" in category or "chat" in category:
            return "chitchat"
        elif "general" in category:
            return "general"
        else:
            return "general"  # fallback
    except Exception as e:
        print("UNIVERRA Classifier Error:", e)
        return "general"
