from typing import Union, List, Optional, Dict
import oss2
from fastapi import FastAPI
from pydantic import BaseModel
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

# Example usage remains the same
# chatInputParams = {
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Who won the world series in 2020?"},
#         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#         {"role": "user", "content": "Where was it played?"}
#     ],
#     "document_id": None
# }