from typing import Union, List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import PyPDFLoader
import uuid
from uuid import uuid4
import oss2
import os
import getpass
import fitz  # PyMuPDF
from dotenv import load_dotenv
from io import BytesIO
from langchain.schema import Document
from fastapi.middleware.cors import CORSMiddleware
import json

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

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    
access_key_id = os.environ.get('OSS_KEY_ID')
access_key_secret = os.environ.get('OSS_KEY_SECRET')
bucket_name = os.environ.get('OSS_BUCKET')
endpoint = os.environ.get('OSS_TEST_ENDPOINT')




app = FastAPI()
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    messages: List[ChatMessage]
    document_id: str   # Optional document ID for context from vector DB

class SourceLocation(BaseModel):
    page_number: int
    position: Dict[str, float] = None  # x, y coordinates in the PDF
    text_snippet: str

class Subheader(BaseModel):
    title: str
    description: str
    sources: List[int]

class Section(BaseModel):
    header: str
    subheader: List[Subheader]

class ResponseStructure(BaseModel):
    section: List[Section]


section : [
    # NODE 1
    {
        "header": "What is the definition of a vector?",
        # SUPPORT POINTS
        "subheader": [
            {
                "title": "Definition",
                "description": "A vector is a mathematical object that has both magnitude and direction. It is represented by an arrow with a length and a direction.",
                "sources": [1, 2, 3]
            },
            {
                "title": "Example",
                "description": "A vector can be represented as an arrow with length 3 and direction pointing to the right.",
                "sources": [4, 5]
            }
        ]
    },
    # NODE 2
     {
        "header": "What is the definition of a vector?",
        # SUPPORT POINTS
        "subheader": [
            {
                "title": "Definition",
                "description": "A vector is a mathematical object that has both magnitude and direction. It is represented by an arrow with a length and a direction.",
                "sources": [1, 2, 3]
            },
            {
                "title": "Example",
                "description": "A vector can be represented as an arrow with length 3 and direction pointing to the right.",
                "sources": [4, 5]
            }
        ]
    }
    
]


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
async def chat(chat_input: ChatInput) -> ResponseStructure:
    # Here you would implement the logic to:
    # 1. Get relevant context from vector DB if document_id is provided
    # 2. Generate response with sources

    # Reuse the embedding and vectorstore
    vectorstore = Chroma(
        collection_name="documents",
        persist_directory="./data/chroma",
        embedding_function=embedding,
    )

    # Define the retriever
    retriever = vectorstore.as_retriever()

    # Get the conversation context with roles
    last_user_message = chat_input.messages[-1].content
    previous_response = chat_input.messages[-2].content if len(chat_input.messages) > 1 else ""
    
    # Combine with role labels for clearer context
    search_query = f"Previous answer: {previous_response}\nFollow-up question: {last_user_message}".strip()
    all_docs = retriever.get_relevant_documents(search_query)
    


    # Filter documents by document_id
    def filter_by_document_id(docs, doc_id):
        return [doc for doc in docs if doc.metadata.get("source") == doc_id]

    relevant_docs = filter_by_document_id(all_docs, chat_input.document_id)

    # Prepare context from filtered documents
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page_number', 'unknown')}]:\n{doc.page_content}"
        for doc in relevant_docs
    ])

    # Prompt and LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert assistant that answers user questions and decomposes them into structured, well-organized subtopics if the topic is complex.

    Each response must follow these rules:
    1. If the question is complex, break it down into 2–5 smaller, manageable components. It can be definition/background that clarifies any terminology or prerequisites, step-by-step process that breaks the solution into clear, sequential steps, why/how it works that explain the reasoning or inner mechanics behind each step , examples that provide real-world or simplified examples, common pitfalls/gotchas the mention mistakes to avoid or important edge cases or summary/takaway that recaps the most important points.
    2. Every answer or smaller manageble component must be structured as a **node** with:
    - "header" – a short title
    - "subheader" – you can have multiple subheaders that explain this answer. Your explaination for each subheader should be in the description part.
    - "description" – this will explain your subheader.
    - "sources" – this will be an array that contains all the page numbers where this information was found.
    3. If you don't know the answer to the question, just say "I don't know".


    Format the entire output as a JSON object in this structure, if there are multiple smaller manageble components, the section can be an array that contains all the smaller manageble components:
    {{
    "section": [{{
        "header": "<main answer title>",
        "subheader": [
        {{
            "title": "<subtopic 1 title>",
            "description": "<detailed explanation of subtopic 1>"
            "sources": [<array of page numbers where this information was found>]
        }},
        {{
            "title": "<subtopic 2 title>",
            "description": "<detailed explanation of subtopic 2>"
            "sources": [<array of page numbers where this information was found>]
        }}
        ]
    }}],
    
    or 
    
    "section": [
        {{
        "header": "<smaller manageble component 1 header>",
        "subheader": [
        {{
            "title": "<subtopic 1 title>",
            "description": "<detailed explanation of subtopic 1>"
            "sources": [<array of page numbers where this information was found>]
        }},
        {{
            "title": "<subtopic 2 title>",
            "description": "<detailed explanation of subtopic 2>"
            "sources": [<array of page numbers where this information was found>]
        }}
        ]
    }},
     {{
        "header": "<smaller manageble component 2 header>",
        "subheader": [
        {{
            "title": "<subtopic 1 title>",
            "description": "<detailed explanation of subtopic 1>"
            "sources": [<array of page numbers where this information was found>]
        }},
        {{
            "title": "<subtopic 2 title>",
            "description": "<detailed explanation of subtopic 2>"
            "sources": [<array of page numbers where this information was found>]
        }}
    ]
    }},
    
    ]
    
    }}

    Do not add any extra commentary outside this JSON.
    """
        ),
        (
            "user",
            "Question: {question}\n\nContext:\n{context}"
        )
    ])
    
    
    chain = prompt | llm
    response = chain.invoke({"question": search_query, "context": context})

    # Extract the content from the response
    content_str = response.content
    # Strip code block formatting
    if content_str.startswith("```json"):
        content_str = content_str.strip("`")       # removes all backticks
        content_str = content_str.split("\n", 1)[1]  # removes the first line
        content_str = content_str.rsplit("\n", 1)[0]  # removes the last line

        # Parse the string as JSON
    content_str = json.loads(content_str)
    return content_str
  
class PDFProcessRequest(BaseModel):
    filename: str

class PDFProcessResponse(BaseModel):
    document_id: str
    success: bool
    
@app.post("/process-pdf")
async def process_pdf(request: PDFProcessRequest) -> PDFProcessResponse:
    try:    
        # Download PDF from OSS path
        bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
        key = request.filename

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
                    "page_number": str(page_num + 1),
                    "source": document_id,
                    "total_pages": str(len(doc)),
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


