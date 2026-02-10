import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from pinecone import Pinecone


# Load environment variables

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# Initialize Pinecone (NEW SDK)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# Embeddings

embeddings = OpenAIEmbeddings()


# Vector Store

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Helper: format documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# LLM

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)


# Prompt

prompt = ChatPromptTemplate.from_template(
    """
You are a medical assistant.
Answer ONLY using the medical context below.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{question}
"""
)


# RAG Chain (LCEL)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# 🩺 Interactive loop for user input

while True:
    user_question = input("\nEnter your medical question (or 'exit' to quit): ")
    if user_question.lower() == "exit":
        print("Goodbye! Stay healthy 🩺")
        break

    response = rag_chain.invoke(user_question)
    print("\n🩺 Answer:\n")
    print(response.content)