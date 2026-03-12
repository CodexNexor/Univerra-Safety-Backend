# api.py
from __future__ import annotations
import logging
import os
import re
from urllib.parse import urlparse
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import markdown as md
import bleach

# Business logic
from query import process_user_query
from location import handle_location_query

# DB helpers / collections
from app.db import (
    get_recent_messages,
    handle_ask,            # persists user + assistant (TEXT only); we patch HTML here
    conversations,         # Mongo collection
    ensure_user_dir,       # for file cleanup
    generate_session_id,   # for /newchat when id is missing
)

# -------------------------
# Config
# -------------------------
MAX_QUERY_LEN = int(os.getenv("MAX_QUERY_LEN", "4000"))
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

app = Flask(__name__)
CORS(app)

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
app.logger.handlers.clear()
app.logger.addHandler(_handler)
app.logger.setLevel(logging.INFO)

# -------------------------
# Markdown → HTML (top-level answer_html only)
# -------------------------
MD_EXTENSIONS = ["extra", "tables", "sane_lists", "nl2br"]

ALLOWED_TAGS = bleach.sanitizer.ALLOWED_TAGS.union({
    "p", "br", "hr", "pre", "code",
    "table", "thead", "tbody", "tr", "th", "td",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "strong", "em", "blockquote", "span", "a"
})
ALLOWED_ATTRS = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "a": ["href", "title", "target", "rel"],
    "span": ["class"],
    "th": ["colspan", "rowspan", "align"],
    "td": ["colspan", "rowspan", "align"]
}
ALLOWED_PROTOCOLS = ["http", "https", "mailto"]

SOURCES_SECTION_PATTERN = re.compile(
    r"(?:\n|\r|^)\s*Sources?:[\s\S]*?(?=(?:\n{2,}|\Z))",
    re.IGNORECASE
)

def strip_plaintext_sources(text: str) -> str:
    return re.sub(SOURCES_SECTION_PATTERN, "", text or "").strip()

def md_to_html_with_sources(md_text: str) -> str:
    url_pattern = re.compile(r"(https?://[^\s)]+)")
    urls = list(dict.fromkeys(url_pattern.findall(md_text or "")))

    html = md.markdown(md_text or "", extensions=MD_EXTENSIONS)
    safe_html = bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=ALLOWED_PROTOCOLS,
        strip=True
    )

    if urls:
        sources_html = "<div style='margin-top:20px; font-size:0.9em;'><strong>Sources:</strong><ul>"
        for url in urls:
            domain = urlparse(url).netloc.replace("www.", "")
            sources_html += f"<li><a href='{url}' target='_blank' rel='noopener noreferrer'>{domain}</a></li>"
        sources_html += "</ul></div>"
        safe_html += sources_html

    return safe_html

# Return HTML only when it's actually *structural*
STRUCTURAL_TAGS_RE = re.compile(r"<(table|tr|td|th|thead|tbody|ul|ol|li|pre|code|blockquote|h[1-6])\b", re.I)
def build_structural_html_or_none(md_text: str) -> str | None:
    """
    Convert markdown to sanitized HTML, but only return it if it contains
    structural elements beyond simple paragraphs. Otherwise return None.
    """
    md_text = (md_text or "").strip()
    if not md_text:
        return None

    html = md.markdown(md_text, extensions=MD_EXTENSIONS)
    safe_html = bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=ALLOWED_PROTOCOLS,
        strip=True
    )
    if STRUCTURAL_TAGS_RE.search(safe_html):
        # When structural, attach a sources block too
        return md_to_html_with_sources(md_text)
    return None

def _text_summary_for_structured(md_text: str, html: str | None) -> str:
    """
    If HTML contains structured content (table/list/code), return only a short
    summary/heading from the markdown to avoid showing the same content twice.
    Strategy: keep text up to the first table/list/code block.
    """
    if not html:
        return md_text

    # Only summarize when HTML actually has structure
    if not STRUCTURAL_TAGS_RE.search(html or ""):
        return md_text

    lines = (md_text or "").splitlines()
    if not lines:
        return md_text

    summary_lines: List[str] = []
    for ln in lines:
        s = ln.strip()
        # Heuristics: stop before table row, list item, or code fence
        if s.startswith("|") or s.startswith("- ") or s.startswith("* ") or s.startswith("```"):
            break
        summary_lines.append(ln)

    summary = "\n".join(summary_lines).strip()
    if not summary:
        # Fallback to first non-empty line
        for ln in lines:
            if ln.strip():
                summary = ln.strip()
                break
    return summary or md_text

# -------------------------
# Helpers
# -------------------------
def _error(status: int, message: str):
    return jsonify({"error": message}), status

def _serialize_history_with_html(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For /history: include answer_html for assistant messages.
    Keys per message: role, text, answer_html, ts
    """
    out = []
    for m in messages or []:
        out.append({
            "role": m.get("role"),
            "text": m.get("text"),
            "answer_html": m.get("html") if m.get("role") == "assistant" else None,
            "ts": m.get("ts"),
        })
    return out

def _compose_email_with_session(email: str, session_id: str | None) -> str:
    """Build 'email/<session_id>' only if session_id exists; else just email."""
    return f"{email}/{session_id}" if session_id else email

def _persist_last_assistant_html(email: str, session_id: str | None, assistant_text: str, assistant_html: str | None) -> None:
    """
    After handle_ask has stored the turn (user + assistant), persist HTML snapshot
    into the most recent assistant message that matches assistant_text.
    """
    if not email or not session_id or not assistant_html:
        return
    try:
        doc = conversations.find_one({"email": email, "session_id": session_id}, {"messages": 1})
        if not doc:
            return
        msgs = doc.get("messages", []) or []
        if not msgs:
            return

        # Find the last assistant message that matches the text we just saved
        idx = None
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i] or {}
            if m.get("role") == "assistant" and (m.get("text") or "") == assistant_text:
                idx = i
                break
        # Fallback: if last message is assistant, patch it
        if idx is None and msgs[-1].get("role") == "assistant":
            idx = len(msgs) - 1
        if idx is None:
            return

        conversations.update_one(
            {"email": email, "session_id": session_id},
            {"$set": {f"messages.{idx}.html": assistant_html}}
        )
    except Exception as e:
        app.logger.warning("Could not persist assistant html: %s", e)

# -------------------------
# Routes
# -------------------------

@app.route("/sessions", methods=["GET"])
def list_sessions():
    """
    Return all sessions (session_id + title) for a given user.
    GET /sessions?email=<user@example.com>
    """
    try:
        email = (request.args.get("email") or "").strip()
        if not email:
            return _error(400, "Missing 'email' parameter")

        docs = conversations.find(
            {"email": email},
            {"session_id": 1, "title": 1, "updated_at": 1}
        ).sort("updated_at", -1)

        sessions = [{"session_id": d["session_id"], "title": d.get("title") or "Untitled Chat"} for d in docs]
        return jsonify({"sessions": sessions}), 200
    except Exception as e:
        app.logger.exception("Error listing sessions: %s", e)
        return _error(500, "Failed to fetch sessions")

@app.route("/history", methods=["GET"])
def get_history():
    """
    Return messages for a specific session, including answer_html for assistant messages.
    GET /history?email=<user@example.com>&session_id=<12digits>[&limit=N]
    """
    try:
        email = (request.args.get("email") or "").strip()
        session_id = (request.args.get("session_id") or "").strip()
        limit_raw = request.args.get("limit")

        if not email or not session_id:
            return _error(400, "Missing 'email' or 'session_id'")

        proj = {"messages": 1, "title": 1, "updated_at": 1, "created_at": 1, "country": 1}
        if limit_raw:
            try:
                n = max(1, int(limit_raw))
            except ValueError:
                return _error(400, "Invalid 'limit' value")
            proj["messages"] = {"$slice": -n}

        doc = conversations.find_one({"email": email, "session_id": session_id}, proj)
        if not doc:
            return _error(404, "Session not found")

        messages = doc.get("messages", [])
        messages = sorted(messages, key=lambda m: m.get("ts")) if messages else []

        return jsonify({
            "email": email,
            "session_id": session_id,
            "title": doc.get("title") or "Untitled Chat",
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "count": len(messages),
            "messages": _serialize_history_with_html(messages),  # includes answer_html for assistants
        }), 200
    except Exception as e:
        app.logger.exception("Error fetching history: %s", e)
        return _error(500, "Failed to fetch history")

@app.route("/newchat", methods=["GET"])
def new_chat():
    """
    Initialize a new chat session.
    GET /newchat?email=<user@example.com>&id=<12digits>&title=<optional>
    - If id is missing/invalid, auto-generates a 12-digit id.
    """
    try:
        email = (request.args.get("email") or "").strip()
        session_id = (request.args.get("id") or "").strip()
        title = (request.args.get("title") or "").strip() or None

        if not email:
            return _error(400, "Missing 'email'")

        if not re.fullmatch(r"\d{12}", session_id or ""):
            session_id = generate_session_id()

        conversations.update_one(
            {"email": email, "session_id": session_id},
            {
                "$setOnInsert": {
                    "email": email,
                    "session_id": session_id,
                    "created_at": None,
                    "messages": [],
                },
                "$set": {
                    "updated_at": None,
                },
                **({"$setOnInsert": {"title": title}} if title else {})
            },
            upsert=True
        )

        return jsonify({
            "email": email,
            "session_id": session_id,
            "title": title or "Untitled Chat"
        }), 200
    except Exception as e:
        app.logger.exception("Error creating new chat: %s", e)
        return _error(500, "Failed to create chat")

@app.route("/deletechat", methods=["GET"])
def delete_chat():
    """
    Delete a chat session and its JSON file.
    GET /deletechat?email=<user@example.com>&id=<12digits>
    """
    try:
        email = (request.args.get("email") or "").strip()
        session_id = (request.args.get("id") or "").strip()

        if not email or not session_id:
            return _error(400, "Missing 'email' or 'id'")

        res = conversations.delete_one({"email": email, "session_id": session_id})
        # Try removing local json snapshot if present
        try:
            user_dir = ensure_user_dir(email)
            fpath = os.path.join(user_dir, f"{session_id}.json")
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception:
            pass

        return jsonify({
            "email": email,
            "session_id": session_id,
            "deleted": bool(res.deleted_count)
        }), 200
    except Exception as e:
        app.logger.exception("Error deleting chat: %s", e)
        return _error(500, "Failed to delete chat")

@app.route("/store", methods=["GET"])
def store_turn():
    """
    Persist a turn WITHOUT running the model.
    GET /store?email=<email>&session_id=<12digits>&query=...&answer=...[&country=in]

    Response (minimal - NO history):
      { answer, answer_html (or null), deep_memory, session_id, email }
    """
    try:
        email = (request.args.get("email") or "").strip()
        session_id = (request.args.get("session_id") or "").strip() or None
        user_query_raw = (request.args.get("query") or "").strip()
        answer_text = (request.args.get("answer") or "").strip()
        country = (request.args.get("country") or "").strip() or None

        if not email:
            return _error(400, "Missing 'email'")
        if not user_query_raw:
            return _error(400, "Missing 'query'")
        if not answer_text:
            return _error(400, "Missing 'answer'")
        if len(user_query_raw) > MAX_QUERY_LEN or len(answer_text) > MAX_QUERY_LEN:
            return _error(413, "Payload too long")

        email_with_session = _compose_email_with_session(email, session_id)

        # Build HTML (only if structural) and reduce text to summary if needed
        cleaned_answer = strip_plaintext_sources(answer_text)
        answer_html = build_structural_html_or_none(cleaned_answer)
        display_text = _text_summary_for_structured(cleaned_answer, answer_html)

        # Persist the turn using the *summary text* (prevents duplication in history/UI)
        result = handle_ask(
            email_and_session=email_with_session,
            query=user_query_raw,
            answer=display_text,
            country=country,
        )
        final_session_id = result.get("session_id", session_id)

        # Patch HTML onto the matching assistant message (use display_text as key)
        if answer_html:
            _persist_last_assistant_html(email, final_session_id, display_text, answer_html)

        return jsonify({
            "answer": display_text,
            "answer_html": answer_html,  # may be None
            "deep_memory": result.get("deep_memory", {}),
            "session_id": final_session_id,
            "email": email
        }), 200

    except Exception as e:
        app.logger.exception("Store Error: %s", e)
        return _error(500, "Failed to store turn")

@app.route("/ask", methods=["GET", "POST"])
def ask_univerra():
    """
    STRICT MODE: requires email, session_id, query, country.
    - If any is missing → 400.
    - If session (email, session_id) does not exist → 404 (create via /newchat first).
    - Never auto-create a session here.
    Returns:
      { answer, answer_html (or null), session_id, email, deep_memory }
    """
    try:
        # Parse inputs (support GET and POST)
        if request.method == "POST":
            if not request.is_json:
                return _error(400, "Body must be JSON")
            data: Dict[str, Any] = request.get_json(silent=True) or {}
            user_query_raw = (data.get("query") or "").strip()
            email = (data.get("email") or "").strip()
            session_id = (data.get("session_id") or "").strip()
            country = (data.get("country") or "").strip()
            location_flag = str(data.get("location", "false")).lower() == "true"
        else:
            user_query_raw = (request.args.get("query") or "").strip()
            email = (request.args.get("email") or "").strip()
            session_id = (request.args.get("session_id") or "").strip()
            country = (request.args.get("country") or "").strip()
            location_flag = str(request.args.get("location", "false")).lower() == "true"

        # Mandatory checks
        if not email:
            return _error(400, "Missing 'email'")
        if not session_id:
            return _error(400, "Missing 'session_id' (create one via /newchat)")
        if not user_query_raw:
            return _error(400, "Missing 'query'")
        if not country:
            return _error(400, "Missing 'country'")

        if len(user_query_raw) > MAX_QUERY_LEN:
            return _error(413, "Query too long")

        # Ensure session exists (no auto-create here)
        exists = conversations.find_one({"email": email, "session_id": session_id}, {"_id": 1})
        if not exists:
            return _error(404, "Session not found. Create it first with /newchat.")

        # Build last 5 messages for this session
        history = get_recent_messages(email, session_id, limit=5)

        # Model input (optional country hint)
        user_query_for_model = f"{user_query_raw} (country={country})" if country else user_query_raw

        # Route to appropriate pipeline
        if location_flag:
            response = handle_location_query(user_query_for_model)
        else:
            response = process_user_query(user_query_for_model, history=history)

        # Normalize output
        if isinstance(response, str):
            answer_text = response
        elif isinstance(response, dict):
            answer_text = response.get("answer", "")
        else:
            answer_text = str(response)

        # Prepare HTML/summary to prevent duplicate rendering
        cleaned_answer = strip_plaintext_sources(answer_text)
        answer_html = build_structural_html_or_none(cleaned_answer)
        display_text = _text_summary_for_structured(cleaned_answer, answer_html)

        # Persist the turn into this existing session using the *summary text*
        email_with_session = _compose_email_with_session(email, session_id)
        result = handle_ask(
            email_and_session=email_with_session,
            query=user_query_raw,
            answer=display_text,
            country=country,
        )
        final_session_id = result.get("session_id", session_id)
        deep_memory = result.get("deep_memory", {})

        # Patch HTML onto the matching assistant message (use display_text as key)
        if answer_html:
            _persist_last_assistant_html(email, final_session_id, display_text, answer_html)

        return jsonify({
            "answer": display_text,
            "answer_html": answer_html,  # may be None
            "deep_memory": deep_memory,
            "session_id": final_session_id,
            "email": email
        }), 200

    except Exception as e:
        app.logger.exception("API Error: %s", e)
        return _error(500, "Something went wrong processing the query")

# -------------------------
# Error handler
# -------------------------
@app.errorhandler(HTTPException)
def handle_http_exception(err: HTTPException):
    return _error(err.code or 500, err.description or err.name or "HTTP error")

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=DEBUG)
