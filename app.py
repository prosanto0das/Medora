from flask import Flask, render_template, request, session, Response, stream_with_context

from src.helper import get_embeddings

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from src.prompt import system_prompt

import os
import re
import sqlite3
import uuid


app = Flask(__name__)

load_dotenv()

# Required for Flask sessions (used for conversation memory)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-me")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Embedding model
embeddings = get_embeddings()


# Pinecone
index_name = "medora"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# LLM
model = ChatOpenAI(
    model="aws-bedrock/anthropic.claude-sonnet-4-6",
    api_key=os.getenv("COMPANY_API_KEY"),
    base_url=os.getenv("COMPANY_BASE_URL"),
    temperature=0.7,
    max_tokens=100,
    streaming=True
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# ---------- Conversation memory (SQLite) ----------
# Stores the last few messages for each browser session in a file,
# so history survives server restarts.
DB_PATH = "chat_history.db"

# Keep only the last 12 messages (6 exchanges) to save tokens
MAX_HISTORY_MESSAGES = 12

# check_same_thread=False lets Flask's threads share one connection
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.execute(
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    )
    """
)
_db.commit()


def get_history(session_id):
    """Return the last MAX_HISTORY_MESSAGES messages for a session."""
    rows = _db.execute(
        """
        SELECT role, content FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, MAX_HISTORY_MESSAGES),
    ).fetchall()
    # Reverse so the oldest kept message comes first
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def save_message(session_id, role, content):
    """Insert one message into the database."""
    _db.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    _db.commit()


def trim_history(session_id):
    """Delete old rows so only the last MAX_HISTORY_MESSAGES stay in the DB."""
    _db.execute(
        """
        DELETE FROM messages
        WHERE session_id = ?
          AND id NOT IN (
              SELECT id FROM messages
              WHERE session_id = ?
              ORDER BY id DESC
              LIMIT ?
          )
        """,
        (session_id, session_id, MAX_HISTORY_MESSAGES),
    )
    _db.commit()


def build_history_text(messages):
    """Turn stored messages into a short text block for the prompt."""
    lines = []
    for message in messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def format_context(docs):
    """Join retrieved documents into one clean text block."""
    return "\n\n".join(doc.page_content for doc in docs)


def clean_chunk(chunk):
    """Light cleaning for streaming chunks.

    No .strip() here — chunks often start with a space that separates
    words, and stripping it would merge words together.
    """
    # Remove <function_calls>...</function_calls> blocks (Anthropic tool-use format)
    cleaned = re.sub(r"<function_calls>.*?</function_calls>", "", chunk, flags=re.DOTALL)
    # Remove any leftover <invoke>...</invoke> blocks
    cleaned = re.sub(r"<invoke>.*?</invoke>", "", cleaned, flags=re.DOTALL)
    # Remove any leftover <parameter ...>...</parameter> blocks
    cleaned = re.sub(r"<parameter[^>]*>.*?</parameter>", "", cleaned, flags=re.DOTALL)
    # Convert literal \n, \r\n, \\n into real line breaks
    cleaned = (
        cleaned
        .replace("\\\\r\\\\n", "\n")
        .replace("\\\\n", "\n")
        .replace("\\r\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\n", "\n")
    )
    return cleaned


def clean_response(response):
    """Full cleaning for a complete response (used when saving to the DB)."""
    cleaned = clean_chunk(response)
    # Collapse extra blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# RAG chain
chain = (   
    {
        "context": RunnableLambda(lambda x: format_context(retriever.invoke(x["input"]))),
        "history": RunnableLambda(lambda x: x["history"]),
        "input": RunnableLambda(lambda x: x["input"]),
    }
    | prompt
    | model
    | StrOutputParser()
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    # Give each browser a unique ID (stored in the session cookie)
    if "user_id" not in session:
        session["user_id"] = uuid.uuid4().hex
    session_id = session["user_id"]

    # Get this user's past messages (empty list on first visit)
    history = get_history(session_id)

    # Turn the last few messages into plain text for the prompt
    history_text = build_history_text(history)

    print("Question:", msg)

    # Save the user's message right away so it's not lost if streaming fails
    save_message(session_id, "user", msg)

    def generate():
        """Yield cleaned text chunks as the model streams them."""
        full_response = []
        try:
            for chunk in chain.stream({"input": msg, "history": history_text}):
                piece = clean_chunk(chunk)
                if piece:
                    full_response.append(piece)
                    yield piece
        except Exception as error:
            print("Stream error:", error)
            yield "\n\nI'm sorry, something went wrong while generating a response. Please try again."

        # Save the assistant's full response once streaming finishes
        response = clean_response("".join(full_response))
        if not response:
            response = "I'm sorry, I couldn't generate a response. Could you rephrase your question?"
        save_message(session_id, "assistant", response)
        trim_history(session_id)
        print("Response:", response)

    response = Response(stream_with_context(generate()), mimetype="text/plain")
    # Tell proxies/browsers not to buffer the stream
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    return response


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )