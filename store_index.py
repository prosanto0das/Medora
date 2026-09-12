from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_files,
    filter_metadata,
    text_split,
    get_embeddings
)

import os


# Load environment variables
load_dotenv()


# Get Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone index name
index_name = "medora"


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


# Create the index only if it doesn't already exist
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name not in existing_indexes:

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"Index '{index_name}' created successfully!")

else:
    print(f"Index '{index_name}' already exists.")


# Connect to the index
index = pc.Index(index_name)


# Check whether documents already exist
stats = index.describe_index_stats()

vector_count = stats["total_vector_count"]

print("Current vector count:", vector_count)


# Create embeddings
embedding = get_embeddings()


# Upload documents only if the index is empty
if vector_count == 0:

    print("Index is empty. Processing and uploading documents...")

    # Load and process documents
    extracted_data = load_pdf_files("Data")
    filtered_data = filter_metadata(extracted_data)
    chunks = text_split(filtered_data)

    # Create vector store and upload documents
    docsearch = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=index_name
    )

    print("Documents uploaded successfully!")

else:

    print("Documents already exist. Skipping upload.")

    # Connect to existing vector store
    docsearch = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding
    )



