import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import textwrap
import chromadb
from pydantic import BaseModel
import os
import google.generativeai as genai
import glob
from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from utils import convert_html_file_to_document
#embedding model initiation
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")   
genai.configure(api_key=api_key)
for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)
    
# text processing functions
    

html_files = glob.glob("templates/*.html")

# Process each HTML file
for html_file in html_files:
    page_topic = os.path.splitext(os.path.basename(html_file))[0]
    output_text_file = f"knowledge_docs/{page_topic}.txt"
    convert_html_file_to_document(html_file, output_text_file)

knowledge_list = []
# read all the files in the knowledge_docs folder
knowledge_files = glob.glob("knowledge_docs/*.txt")
for i, file in enumerate(knowledge_files):
    with open(file, 'r') as f:
        text = f.read()
    paragraphs = textwrap.wrap(text, width=100)
    for p in paragraphs:
        print(len(p))
        knowledge_list.append({
            'paragraph': p,
            'file': file,
            'page_index': i,
            'paragraph_index': paragraphs.index(p)
        })
        

# Embedding and ChromaDB functions
class DocumentRequest(BaseModel):
    document_id: str
    
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        return genai.embed_content(model=model,
                                    content=input,
                                    task_type="retrieval_document",
                                    )["embedding"]
    
def create_chroma_db(knowledge_list, name):
    client = chromadb.PersistentClient(path="dbs/my_local_data")  # or HttpClient()
    collection = client.get_or_create_collection(name, embedding_function=GeminiEmbeddingFunction())
    upper_bound = len(knowledge_list)
    collection.add(
        ids=[str(i) for i in range(upper_bound)],
        documents=[knowledge_list[i]["paragraph"] for i in range(upper_bound)],
        metadatas=[
            {"name": knowledge_list[i]["file"],
             "page_index": knowledge_list[i]["page_index"],
             "paragraph_index": knowledge_list[i]["paragraph_index"]} 
            for i in range(upper_bound)],
    )
    return collection

# Set up the DB
db = create_chroma_db(knowledge_list, "website_knowledge")
print(db)
# print(dir(GeminiEmbeddingFunction("How are you doing today?")))

