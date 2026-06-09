from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import chromadb
import ollama

Tk().withdraw()

pdf_path = askopenfilename(
    filetypes = [("PDF Files", "*.pdf")]
)
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    info = page.extract_text()

    if info:
        text += info
#Check to ensure text is being extracted correctly

#Chunk the text into single sentances for smaller pieces for embedding
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
query = input("What do you want to search for in the document? ")
#Embed it using the same model and pass it to collection.query()
query_embedding = model.encode(query).tolist()

#Query the vector store 
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

all_results = ""
for i in range(5):
    all_results += results['documents'][0][i] + '\n'

#Send Ollama the query and the relevant chunks to get a response    
ollama_query = f"You are a research assistant. The context from the document is: {all_results}. The user wants to know: {query}. Based on the information provided, answer the user's question as best as you can. If you don't know the answer, say you don't know."
response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": ollama_query}])
print(response['message']['content'])