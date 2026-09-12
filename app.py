from flask import Flask, render_template, request

from src.helper import get_embeddings

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from src.prompt import system_prompt

import os


app = Flask(__name__)

load_dotenv()

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


# RAG chain
chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough()
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

    print("Question:", msg)

    response = chain.invoke(msg)

    print("Response:", response)

    return response


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )