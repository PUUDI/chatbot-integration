from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
import os
import asyncio

app = FastAPI()

#embedding model initiation
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")   
genai.configure(api_key=api_key)

class DocumentRequest(BaseModel):
    document_id: str
    
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        return genai.embed_content(model=model,
                                    content=input,
                                    task_type="retrieval_document",
                                    )["embedding"]

@app.get("/retrieve_document/{document_id}")
# async def retrieve_document(document_id: str):
def retrieve_document(document_id: str):
    try:
        # Initialize ChromaDB client
        client = chromadb.Client()

        # Connect to the local database
        db = client.connect("dbs/my_local_data")

        # Retrieve the document by ID
        document = db.get_document(document_id)

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"document_id": document_id, "content": document.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query_document")
def query_document(text: str):
    try:
        # Initialize ChromaDB client
        client = chromadb.Client()

        # Connect to the local database
        # Load or connect to an existing collection
        client = chromadb.PersistentClient(path="dbs/my_local_data")
        collection = client.get_collection("website_knowledge", GeminiEmbeddingFunction())
        
        result = collection.query(
        query_texts=[text],
        n_results=3  # Number of similar documents to retrieve
        )

        if not result:
            raise HTTPException(status_code=404, detail="No similar documents found")

        return {"query_text": text, "results": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
    
    
    # #test
    # response = query_document("How to create a new thread in Python?")
    # print(response)