RAG Research Assistant

An AI-powered research assistant that lets you upload a PDF and ask questions about it in natural language. Built on a full Retrieval-Augmented Generation (RAG) pipeline — no hallucinations from training data, just answers grounded in your document.

## Description

This project was initially devised for students to test themselves on the subject matter that they study. By uploading a PDF of what they're studying (e.g. a textbook, mock exam, or problem set) students can query an AI agent to explain unfamiliar concepts or bounce questions off of it.

## Features

- Sentence-level chunking with NLTK
- Local embedding generation via Sentence Transformers (384-dimension vectors)
- Semantic similarity search using a ChromaDB vector store
- Used the Groq LLM API to generate answers from retrieved content
- Shipped the app to production on Streamlit Cloud

## Installation

### Prerequisites
- Node.js v14+
- npm or yarn

### Steps

```bash
git clone https://github.com/username/project.git
cd project
npm install
