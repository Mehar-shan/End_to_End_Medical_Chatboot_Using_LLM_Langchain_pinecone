from src.helper import ( load_pdf_file, text_split,
                         download_huggingface_model)    
import os
from dotenv import load_dotenv
load_dotenv()

import pinecone  # keep for general use
from pinecone import Pinecone, ServerlessSpec  # new client
from langchain_pinecone import Pinecone as LangChainPinecone  # LangChain wrapper

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Extract and process PDF data
extracted_Data = load_pdf_file('Data/')
text_chunks = text_split(extracted_Data)
embeddings = download_huggingface_model()

# Create Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Create index if it does not exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,   # sentence-transformer size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Create LangChain Pinecone vectorstore and upsert chunks
docsearch = LangChainPinecone.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
