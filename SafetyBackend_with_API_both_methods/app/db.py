import os
import re
import json
import errno
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError
from groq import Groq

# =========================
# Environment & Constants
# =========================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "univerra")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USER_DATA_DIR = os.getenv("USER_DATA_DIR", "./user_data")  # where per-email folders live

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment")

# =========================
# MongoDB Client & Collections
# =========================
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collection storing per (email, session_id) conversation docs
conversations = db["conversations"]
users = db["users"]

# Indexes
try:
    conversations.drop_index("email_1")
    conversations.create_index([("email", ASCENDING), ("session_id", ASCENDING)], unique=True, background=True)
    conversations.create_index([("email", ASCENDING)], background=True)
    users.create_index([("email", ASCENDING)], unique=True, background=True)
except Exception as e:
    import logging
    logging.warning("Index creation skipped: %s", e)

# =========================
# Groq Client (Memory Extraction) Setup
# =========================
groq_client = Groq(api_key=GROQ_API_KEY)

# =========================
# Utilities for File and Directory Management
# =========================
SAFE_EMAIL_CHARS = re.compile(r"[^a-zA-Z0-9_.@+-]")

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _sanitize_email_for_fs(email: str) -> str:
    cleaned = SAFE_EMAIL_CHARS.sub("_", email.strip())
    return cleaned[:128] if len(cleaned) > 128 else cleaned

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def ensure_user_dir(email: str) -> str:
    base = os.path.abspath(USER_DATA_DIR)
    _ensure_dir(base)
    folder = os.path.join(base, _sanitize_email_for_fs(email))
    _ensure_dir(folder)
    return folder

def generate_session_id() -> str:
    import random
    return "".join(str(random.randint(0, 9)) for _ in range(12))

def _session_file_path(email: str, session_id: str) -> str:
    user_dir = ensure_user_dir(email)
    return os.path.join(user_dir, f"{session_id}.json")

def _memory_file_path(email: str) -> str:
    user_dir = ensure_user_dir(email)
    return os.path.join(user_dir, "memory.json")

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)

def _extract_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}

# =========================
# User Registry Helpers
# =========================
def ensure_user(email: str) -> None:
    ensure_user_dir(email)
    now = _utcnow()
    try:
        users.update_one(
            {"email": email},
            {"$setOnInsert": {"email": email, "created_at": now}, "$set": {"updated_at": now}},
            upsert=True,
        )
    except DuplicateKeyError:
        pass

# =========================
# Deep Memory Handling
# =========================
DEEP_MEMORY_SYSTEM = "You extract persistent user facts in strict JSON."
DEEP_MEMORY_INSTRUCTIONS = """
You are a memory extraction system.
From the text below, extract ONLY persistent, non-ephemeral facts about the user.
Prefer keys like: name, email, phone, dob, city, country, timezone, likes, dislikes, preferences, goals, interests, profession, company, language, etc.
Output STRICT JSON with a flat object or shallow objects (no prose, no markdown).
If nothing solid is found, output {}.

Text:
"""

def _groq_deep_memory(text: str) -> Dict[str, Any]:
    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": DEEP_MEMORY_SYSTEM},
                {"role": "user", "content": DEEP_MEMORY_INSTRUCTIONS + text},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        import logging
        logging.warning("Deep memory LLM error: %s", e)
    return {}

def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst or {})
    for k, v in (src or {}).items():
        out[k] = v
    return out

def process_and_store_deep_memory(email: str, candidate_texts: List[str]) -> Dict[str, Any]:
    ensure_user(email)
    merged: Dict[str, Any] = {}

    for txt in candidate_texts:
        if not txt or not txt.strip():
            continue
        extracted = _groq_deep_memory(txt.strip())
        if extracted:
            merged = _merge_dict(merged, extracted)

    if not merged:
        doc = conversations.find_one({"email": email}, {"deep_memory": 1})
        current = (doc or {}).get("deep_memory", {})
        return current or {}

    conversations.update_many(
        {"email": email},
        {"$set": {f"deep_memory.{k}": v for k, v in merged.items()}, "$setOnInsert": {"created_at": _utcnow()}},
        upsert=True,
    )

    mem_path = _memory_file_path(email)
    existing = _read_json(mem_path) or {}
    combined = _merge_dict(existing, merged)
    _write_json(mem_path, combined)

    return combined

def get_deep_memory(email: str) -> Dict[str, Any]:
    doc = conversations.find_one({"email": email}, {"deep_memory": 1})
    if doc and "deep_memory" in doc:
        return doc["deep_memory"]
    mem = _read_json(_memory_file_path(email)) or {}
    return mem

# =========================
# Conversation Handling
# =========================
def _init_or_get_session(
    email: str,
    session_id: Optional[str],
    country: Optional[str],
    first_user_message: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    ensure_user(email)
    sid = session_id or generate_session_id()
    now = _utcnow()

    title = (first_user_message or "").strip() or None

    update = {
        "$setOnInsert": {
            "email": email,
            "session_id": sid,
            "created_at": now,
            "messages": [],
        },
        "$set": {
            "updated_at": now,
        },
    }
    if country:
        update["$set"]["country"] = country
    if title:
        update["$setOnInsert"]["title"] = title

    doc = conversations.find_one_and_update(
        {"email": email, "session_id": sid},
        update,
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return sid, doc

def append_message(
    email: str,
    session_id: str,
    role: str,
    text: str,
    country: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_user(email)
    now = _utcnow()
    message = {"role": role, "text": text, "ts": now}

    update = {
        "$push": {"messages": message},
        "$set": {"updated_at": now},
    }
    if country:
        update["$set"]["country"] = country

    doc = conversations.find_one_and_update(
        {"email": email, "session_id": session_id},
        update,
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    _persist_session_file(email, doc)
    return doc

def _persist_session_file(email: str, conv_doc: Dict[str, Any]) -> None:
    session_id = conv_doc["session_id"]
    user_dir = ensure_user_dir(email)
    session_path = os.path.join(user_dir, f"{session_id}.json")

    payload = {
        "email": email,
        "session_id": session_id,
        "country": conv_doc.get("country"),
        "title": conv_doc.get("title"),
        "created_at": conv_doc.get("created_at"),
        "updated_at": conv_doc.get("updated_at"),
        "messages": conv_doc.get("messages", []),
        "deep_memory_snapshot": conv_doc.get("deep_memory", get_deep_memory(email)),
    }
    _write_json(session_path, payload)

def get_recent_messages(email: str, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    doc = conversations.find_one(
        {"email": email, "session_id": session_id},
        {"messages": {"$slice": -abs(limit)}},
    )
    if not doc:
        return []
    return doc.get("messages", [])

def save_qa_pair(
    email: str,
    session_id: Optional[str],
    user_query: str,
    ai_answer: str,
    country: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    sid, _doc = _init_or_get_session(
        email=email,
        session_id=session_id,
        country=country,
        first_user_message=user_query,
    )

    upsert_session_title(email, sid, user_query)

    doc = append_message(email, sid, role="user", text=user_query, country=country)
    doc = append_message(email, sid, role="assistant", text=ai_answer, country=country)

    merged_mem = process_and_store_deep_memory(email, [user_query, ai_answer])

    doc = conversations.find_one_and_update(
        {"email": email, "session_id": sid},
        {"$set": {"deep_memory": merged_mem, "updated_at": _utcnow()}},
        return_document=ReturnDocument.AFTER,
    )
    _persist_session_file(email, doc)

    return sid, doc

def handle_ask(
    email_and_session: str,
    query: str,
    answer: str,
    country: Optional[str] = None,
) -> Dict[str, Any]:
    email, session_id, user_query, cty = parse_ask_params(email_and_session, query, country)
    
    sid, doc = save_qa_pair(
        email=email,
        session_id=session_id,
        user_query=user_query,
        ai_answer=answer,
        country=cty,
    )

    return {
        "email": email,
        "session_id": sid,
        "title": doc.get("title"),
        "updated_at": doc.get("updated_at"),
        "messages_count": len(doc.get("messages", [])),
        "deep_memory": doc.get("deep_memory", {}),
    }

def parse_ask_params(
    email_and_session: str,
    query: str,
    country: Optional[str] = None,
) -> Tuple[str, Optional[str], str, Optional[str]]:
    """
    Accepts the path segment like:
      email=someone@example.com/624374281434
    or just:
      email=someone@example.com
    Returns (email, session_id_or_none, query, country)
    """
    parts = email_and_session.split("/", 1)
    email = parts[0].strip()

    sid: Optional[str] = None
    if len(parts) == 2:
        candidate = parts[1].strip()
        if re.fullmatch(r"\d{12}", candidate):
            sid = candidate

    return email, sid, query, (country or None)

def upsert_session_title(email: str, session_id: str, candidate_title: Optional[str]) -> None:
    """
    If the session has no title yet and a candidate title is provided (usually first user message),
    set it.
    """
    if not candidate_title:
        return
    conversations.update_one(
        {"email": email, "session_id": session_id, "title": {"$exists": False}},
        {"$set": {"title": candidate_title.strip(), "updated_at": _utcnow()}}
    )
