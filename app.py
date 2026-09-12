from flask import Flask, render_template, request, session

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
    max_tokens=100
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# ---------- Conversation memory (in-memory) ----------
# Stores the last few messages for each browser session.
# Format: {session_id: [{"role": "user"/"assistant", "content": "..."}, ...]}
conversations = {}

# Keep only the last 12 messages (6 exchanges) to save tokens
MAX_HISTORY_MESSAGES = 12


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


def clean_response(response):
    """Remove tool-call XML blocks that sometimes leak into the response."""
    # Remove <function_calls>...</function_calls> blocks (Anthropic tool-use format)
    cleaned = re.sub(r"<function_calls>.*?</function_calls>", "", response, flags=re.DOTALL)
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
    history = conversations.get(session_id, [])

    # Turn the last few messages into plain text for the prompt
    history_text = build_history_text(history)

    print("Question:", msg)

    # Call the RAG chain with the question AND the conversation history
    response = chain.invoke({"input": msg, "history": history_text})

    # Remove tool-call XML that sometimes leaks into the response
    response = clean_response(response)

    # Fallback if the model only returned a tool call
    if not response:
        response = "I'm sorry, I couldn't generate a response. Could you rephrase your question?"

    print("Response:", response)

    # Save this exchange so the next question has context
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": response})
    conversations[session_id] = history[-MAX_HISTORY_MESSAGES:]

    return response


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )