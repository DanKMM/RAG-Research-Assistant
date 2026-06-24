# RAG Research Assistant

An AI-powered research assistant that lets you upload a PDF and ask questions about it in natural language. Built on a full Retrieval-Augmented Generation (RAG) pipeline. No hallucinations from training data, just answers coming straight from your document.

**[Try it live on Streamlit](https://rag-research-assistant-qwlfdwbybmtqm2fekq86s2.streamlit.app/)**

---

## What it does

Upload any PDF, type in a question, and get a response sourced from the document's content. Works great whether the PDF is a research paper, textbook chapter, or mock exam. This project was initially made for students to test themselves by uploading documents and asking questions to improve their own understanding of the subject. 

---

## Features

The app runs a full RAG pipeline through:
- **Extraction** - Text from the PDF is extracted using pypdf
- **Chunking** -  Text is split into individual sentences using NLTK, keeping semantic portions of text intact
- **Embedding** - Each sentence is converted to a 384-dimension vector via Sentence Transformers
- **Storing** - Each vector is then stored in a ChromaDB vector database.
- **Retrieval** - User query is embedded with the same model as the text. The query is then compared to the chunked vectors using cosine similarity and the 5 most similar chunks are returned.
- **Generate** -  Retrieved chunks are sent to Groq's llama3-70b-8192 model to generate answers from the retrieved content.

---

## Tech stack
 
| Layer | Tool |
|---|---|
| PDF extraction | `pypdf` |
| Sentence tokenization | `nltk` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | `ChromaDB` |
| LLM | Groq API (LLaMA 3 70B) |
| Frontend | `Streamlit` |

---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/DanKMM/RAG-Research-Assistant.git
cd RAG-Research-Assistant
```
 
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
 
**3. Set up your Groq API key**
 
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```
 
Get a free API key at [console.groq.com](https://console.groq.com).
 
**4. Run the app**
```bash
python -m streamlit run "PDF Analyzer.py"
```
At the end, your project structure should look like this:
```
RAG-Research-Assistant/
├── PDF Analyzer.py       # Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # API key (DO NOT COMMIT)
└── README.md
```
---

### Why this project?

Standard LLMs like ChatGPT and Claude answer your questions based on their training data and, when pushed to answer a question outside their knowledge base, can hallucinate entirely false answers. However, through a RAG system, the model retrieves relevant information from the documents YOU upload. In other words, the model can only work with what you give it. Your own personal research assistant.


git clone https://github.com/username/project.git
cd project
npm install
