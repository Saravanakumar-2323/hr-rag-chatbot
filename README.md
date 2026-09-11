# 🤖 HR Policy Assistant

An AI-powered HR Policy Chatbot built using Retrieval-Augmented Generation (RAG).

The chatbot retrieves relevant information from the HR Policy PDF and generates grounded answers based only on the provided policy content.

## 🚀 Live Demo

[Try the HR Policy Assistant](https://hr-rag-chatbot-8kwh6lwntkrxhzkrhas5hx.streamlit.app/)

## 🧠 Features

- 📄 PDF-based HR policy question answering
- 🔍 Semantic search using embeddings
- 🗃️ ChromaDB vector database
- 🤖 FLAN-T5 language model
- 🛡️ Hallucination protection
- 📚 Displays retrieved source context
- 🌐 Deployed using Streamlit Community Cloud

## 🔄 RAG Pipeline

PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Retrieval → LLM → Grounded Answer

## 🛠️ Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB
- Hugging Face Transformers
- FLAN-T5

## 📋 Example Questions

- How many casual leaves can an employee take?
- How many sick leaves can an employee take?
- What are the working hours?
- Can employees work from home?
- What is the maternity leave policy?

## 🛡️ Hallucination Prevention

The chatbot uses a similarity threshold and a grounded prompt.

If relevant information is not found in the provided HR policy, the chatbot responds:

> The information is not available in the provided HR policy.

## 📁 Project Structure

```text
hr-rag-chatbot/
│
├── app.py
├── HR_Policy.pdf
├── requirements.txt
└── README.md
