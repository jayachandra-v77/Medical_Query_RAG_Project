#Medical Query RAG Project

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions and receive answers directly from a medical PDF document.

The application uses LangChain, OpenAI, and Pinecone to retrieve relevant content from the document and generate context-based answers.

Project Overview
-------------------------------------------------------------------

> Loads a medical PDF file
> Splits text into smaller chunks
> Converts text into vector embeddings using OpenAI
> Stores embeddings in Pinecone
> Retrieves relevant content for a user query
> Generates answers strictly from the retrieved context

Tech Stack
---------------------------------------------------------------------
Python
VS code
Git
LangChain 
OpenAI (LLM and embeddings)
Pinecone (Vector Database)

Project Structure
---------------------------------------------------------------------
medical-pdf-rag/
│
├── data/
│   └── Medical_book.pdf
├── ingest.py
├── query.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md



Explination:
----------------------------------------------------------------
Example question:

What are the causes and symptoms of diabetes?
How It Works
The PDF content is converted into embeddings
Pinecone retrieves the most relevant sections
OpenAI generates an answer using only the retrieved content

Disclaimer

This project is for educational purposes only and does not provide medical advice.

Author
Jay
