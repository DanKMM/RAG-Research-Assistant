from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')

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

#Chunk the text into smaller pieces for embedding
sentences = sent_tokenize(text)
chunks = []
for i in range(0,len(sentences), 5):
    chunk = ' '.join(sentences[i:i+5])
    chunks.append(chunk)

#Choose a model to embed the chunks
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks)