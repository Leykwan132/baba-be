from typing import Union
import chromadb
from fastapi import FastAPI

client = chromadb.PersistentClient()
app = FastAPI()

@app.post("/process-pdf")
async def process_pdf(file:UploadFile = File):
       tmp_path = None
    try:
        # Save uploaded file
        tmp_path = await save_upload_file(file)
        
        # Process PDF content
        metadata, text_content, pages = process_pdf_content(tmp_path)
        
        # Store in vector database
        vector_store_location = store_in_vectordb(pages)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": "PDF processed successfully",
            "metadata": metadata,
            "num_pages": len(pages),
            "vector_store_location": vector_store_location
        }
    except Exception as e:
        # Clean up temporary file in case of error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}