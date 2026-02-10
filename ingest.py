import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# Load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{INDEX_NAME}' created")
else:
    print(f"Index '{INDEX_NAME}' already exists")

# Load PDF
loader = PyPDFLoader("data/Medical_book.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

# Store in Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Medical PDF successfully ingested into Pinecone")
