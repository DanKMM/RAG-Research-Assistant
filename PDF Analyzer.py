from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import chromadb

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
sentences = sent_tokenize(text)
chunks = []
for sentence in sentences:
    chunks.append(sentence)

#Choose a model to embed the chunks
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks)


#Vector Store with ChromaDB
client = chromadb.PersistentClient(path="./vectorstore")
collection = client.get_or_create_collection(name="research_docs")

#When the user puts in a new file, delete the old collection and create a new one
client.delete_collection(name="research_docs")
collection = client.get_or_create_collection(name="research_docs")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)

print(f"Stored {len(chunks)} chunks in the vector store.")
#Take a text input from the user
query = input("What do you want to search for in the document? ")
#Embed it using the same model and pass it to collection.query()
query_embedding = model.encode(query).tolist()

#Query the vector store 
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
#Print the returned chunks so you can verify it's working
print("Top 5 relevant chunks:") 
for i in results['documents'][0]:
    print(i + "\n") 
