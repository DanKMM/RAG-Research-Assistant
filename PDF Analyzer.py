import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import chromadb
from dotenv import load_dotenv
import os
from groq import Groq

#Load up the Groq API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=api_key)

#Upload file from PDF and extract the text from it
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        info = page.extract_text()

        if info:
            text += info

    # Chunk the text into single sentances for smaller pieces for embedding
    # We need to tokenize the text because simply searching for the plain text meaning will only get us looking at those exact words. 
    # Instead, by tokenizing and embedding, we can look for actual meaning rather than simply looking for the same words. 
    # This is because the embedding will capture the meaning of the sentence, so even if the user searches for something that isn't exactly 
    # the same as the text in the document, it can still return relevant results based on the meaning of the sentences.
    sentences = sent_tokenize(text) 
    chunks = []
    for sentence in sentences:
        chunks.append(sentence)

    #Choose a model to embed the chunks
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)


    #Vector Store with ChromaDB
    client = chromadb.PersistentClient(path="./vectorstore")

    #When the user puts in a new file, delete the old collection and create a new one
    client.delete_collection(name="research_docs")
    collection = client.get_or_create_collection(name="research_docs")

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))]
    )

    #Take a text input from the user
    query = st.text_input("What do you want to search for in the document? ")
    #Embed it using the same model and pass it to collection.query()
    if query:
        query_embedding = model.encode(query).tolist()

        #Query the vector store 
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        all_results = ""
        for i in range(5):
            all_results += results['documents'][0][i] + '\n'
        ollama_query = f"Based on the following information from the document, answer the question: {query}\n\n{all_results}"
        #Create Groq query to send to the LLM
        response = client_groq.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": ollama_query}]
        )
        st.write(response.choices[0].message.content)