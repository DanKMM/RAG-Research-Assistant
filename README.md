# Project Name

This Streamlit-hosted RAG system allows users to upload a PDF file and ask questions based on the PDF file.

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
