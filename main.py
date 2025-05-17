from typing import Union, List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uuid
import oss2
import os
import getpass
import fitz  # PyMuPDF
from dotenv import load_dotenv
from io import BytesIO
from langchain.schema import Document

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
access_key_id = os.environ.get('OSS_KEY_ID')
access_key_secret = os.environ.get('OSS_KEY_SECRET')
bucket_name = os.environ.get('OSS_BUCKET')
endpoint = os.environ.get('OSS_TEST_ENDPOINT')

from dotenv import load_dotenv

# Vector store and embeddings
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_chroma import Chroma
from uuid import uuid4
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LLM and conversation chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings


app = FastAPI()
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post('/chat')
async def chat(chat_input: ChatInput) -> ChatResponse:
    # Here you would implement the logic to:
    # 1. Get relevant context from vector DB if document_id is provided
    # 2. Generate response with sources
    

    # Load environment variables
    load_dotenv()

    pdf_list = [
        {"path": "papers/biology.pdf", "id": "1"},
        {"path": "papers/history.pdf", "id": "2"}
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    all_split_docs = []
    for pdf in pdf_list:
        loader = PyPDFLoader(pdf["path"])
        raw_documents = loader.load()
        documents = [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "pdf_id": pdf["id"]})
            for doc in raw_documents
        ]
        split_docs = text_splitter.split_documents(documents)
        all_split_docs.extend(split_docs)

    uuids = [str(uuid4()) for _ in all_split_docs]
    vector_store.add_documents(all_split_docs, ids=uuids)

    # User question
    query_pdf_id = "1"
    user_prompt = "Who discovered cell?"

    # Prompt and LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert that answers questions based on the provided document context to help students to learn."
            "Return your response in the following JSON format:\n\n"
            "{{\n"
            '  "answer": {{\n'
            '    "Key1": "Description of Key1...",\n'
            '    "Key2": "Description of Key2...",\n'
            '    "Key3": "Description of Key3..."\n'
            "  }},\n"
            "}}\n\n"
            "Only include relevant keys. If the answer is unknown, respond with:\n"
            '{{ "answer": "I don’t know" }}'
        ),
        (
            "user",
            "Question: {question}\nContext: {context}"
        )
    ])

    chain = prompt | llm
    retriever = vector_store.as_retriever()

    def filter_by_pdf_id(docs, pdf_id):
        return [doc for doc in docs if doc.metadata.get("pdf_id") == pdf_id]

    all_docs = retriever.get_relevant_documents(user_prompt)
    docs = filter_by_pdf_id(all_docs, query_pdf_id)

    context = "\n\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"question": user_prompt, "context": context})

    print("Answer:", response.content)
    print("\nTop Sources:")
    for doc in docs:
        print(f"- Page: {doc.metadata.get('page_label', 'unknown')}, PDF ID: {doc.metadata.get('pdf_id')}")
        # print(doc.metadata) 

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

class PDFProcessResponse(BaseModel):
    document_id: str
    success: bool

@app.post("/process-pdf")
async def process_pdf(filename: str) -> PDFProcessResponse:
    try:    
        # Download PDF from OSS path
        bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
        key = filename

        # Get PDF data directly as bytes
        pdf_data = bucket.get_object(key).read()
        try:
            pdf_file = BytesIO(pdf_data)
 
            # Load and process PDF directly from memory
            doc = fitz.open(stream=pdf_file, filetype="pdf")
            # Generate unique document ID
            document_id = str(uuid.uuid4())

            # Process each page and extract metadata
            page_metadata = []
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text with detailed information
                blocks = page.get_text("dict")["blocks"]
                page_text = ""
                coordinates = []
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"]
                                page_text += text + " "
                                coordinates.append({
                                    "x0": span["bbox"][0],
                                    "y0": span["bbox"][1],
                                    "x1": span["bbox"][2],
                                    "y1": span["bbox"][3],
                                    "text": text,
                                    "font": span["font"],
                                    "font_size": span["size"]
                                })
                
                metadata = {
                    "page_number": page_num + 1,
                    "source": document_id,
                    "total_pages": len(doc),
                    # Store coordinates as a string representation or simplified format
                    "coordinate_summary": f"Page {page_num + 1} contains {len(coordinates)} text blocks"
                }
                page_metadata.append(metadata)
                pages_content.append(page_text)
            
            # Split text into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            # Create documents with metadata
            documents = []
            for idx, text in enumerate(pages_content):

                doc_metadata = page_metadata[idx]
                document = Document(
                    page_content=text,
                    metadata=doc_metadata
                )
                documents.append(document)

            splits = text_splitter.split_documents(documents)

            # Store in Chroma with document_id
            vectorstore = Chroma(
                collection_name="documents",  # Single collection for all documents
                persist_directory="./data/chroma",
                embedding_function=embedding,
            )
            
            print(splits)
            # Add documents to vector store with metadata
            vectorstore.add_documents(
                documents=splits,
                ids=[f"{document_id}_{i}" for i in range(len(splits))]
            )

            
            return PDFProcessResponse(
                document_id=document_id,
                success=True
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
                    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing OSS: {str(e)}")