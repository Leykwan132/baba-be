import chromadb
from typing import Union, List, Optional, Dict
import oss2
from fastapi import FastAPI
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from pydantic import BaseModel


client = chromadb.PersistentClient()
app = FastAPI()

# @app.post("/process-pdf")
# async def process_pdf(file:UploadFile = File):
#     tmp_path = None

#     endpoint = 'http://oss-cn-hangzhou.aliyuncs.com' # Suppose that your bucket is in the Hangzhou region.

#     auth = oss2.Auth('<Your AccessKeyID>', '<Your AccessKeySecret>')
#     bucket = oss2.Bucket(auth, endpoint, '<your bucket name>')

#     # The object key in the bucket is story.txt
#     key = 'story.txt'

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    messages: List[ChatMessage]
    document_id: Optional[str] = None  # Optional document ID for context from vector DB

class SourceLocation(BaseModel):
    page_number: int
    position: Dict[str, float] = None  # x, y coordinates in the PDF
    text_snippet: str

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceLocation] = []

@app.post('/chat')
async def chat(chat_input: ChatInput) -> ChatResponse:
    # Here you would implement the logic to:
    # 1. Get relevant context from vector DB if document_id is provided
    # 2. Generate response with sources
    
    # Example response with source information
    return ChatResponse(
        response="The game was played at Globe Life Field in Arlington, Texas",
        sources=[
            SourceLocation(
                page_number=1,
                position={"x": 100, "y": 200},
                text_snippet="The 2020 World Series was played at Globe Life Field..."
            )
        ]
    )

# Example usage remains the same
chatInputParams = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    "document_id": None
}