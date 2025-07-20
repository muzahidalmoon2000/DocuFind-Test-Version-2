import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_intent_and_extract(user_input):
    """
    Detect user intent and extract a clean query for file search if applicable.
    Uses both keyword detection and GPT fallback for better accuracy.
    """
    input_lower = user_input.strip().lower()
    file_keywords = [
        "file", "document", "doc", "pdf", "folder", "record",
        "report", "sheet", "policy", "guide", "manual", "plan", "info"
    ]

    # ✅ Rule-based shortcut for file-related queries
    if any(kw in input_lower for kw in file_keywords):
        match = re.search(
            r"(?:give|send|show|i need|i want|get|find|download|share)\s+(?:me\s+)?(?:the\s+)?(.+?)\s*(?:file|document|folder)?$",
            input_lower
        )
        if match:
            probable_query = match.group(1).strip()
            return {
                "intent": "file_search",
                "data": probable_query
            }


    # ✅ Fallback to GPT for broader understanding
    return detect_intent_and_extract_gpt(user_input)


def detect_intent_and_extract_gpt(user_input):
    """
    Use GPT-4o to classify intent and extract file search keyword(s) in strict JSON.
    """
    system_prompt = (
        "You're an AI assistant for a document assistant application. Your job is to classify user input as either a file search or a general response.\n\n"
        "Reply strictly in JSON format only, like:\n"
        "{\"intent\": \"file_search\", \"data\": \"maternity\"}\n"
        "OR\n"
        "{\"intent\": \"general_response\", \"data\": \"\"}\n\n"
        "Rules:\n"
        "- Use intent 'file_search' if user is trying to get, share, show, download, send, or find a document, info, policy, file, report, or manual.\n"
        "- Extract the clean keyword(s) related to the file — remove filler like: file, document, report, info, etc.\n"
        "- Do not invent keywords. If unclear, return intent as 'general_response'.\n"
        "- Use lowercase unless proper name (e.g., 'Anup').\n"
        "- NEVER return anything except the strict JSON format.\n\n"
        f"User input:\n{user_input}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print("❌ GPT error during intent detection:", e)
        return {"intent": "general_response", "data": ""}


def answer_general_query(user_input):
    """
    Use GPT-4o to respond to non-search general questions or small talk.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are a polite and helpful assistant inside a document search chatbot. "
                    "You can answer greetings, general queries, or ask clarifying questions if needed."
                )},
                {"role": "user", "content": user_input}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT error during general query:", e)
        return "⚠️ I'm having trouble responding. Please try again shortly."
