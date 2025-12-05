# main.py
import os
import tempfile
import json
from typing import List, Union, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains.retrieval import RetrievalQA


# from langchain.document_loaders import PyPDFLoader, JSONLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI

load_dotenv()

# Ensure OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable (do NOT commit keys).")

# Config
OPENAI_MODEL = "gpt-4o-mini"  # per the challenge
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # OpenAI embedding name that works with LangChain - change if needed

app = FastAPI(title="Zania QA API", version="0.1.0")


class QAItem(BaseModel):
    question: str
    answer: str


def load_document_from_file(temp_path: str, filename: str):
    """
    Choose loader based on file extension and return list[Document].
    LangChain Document object is returned by loaders.
    """
    name_lower = filename.lower()
    if name_lower.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        return docs
    elif name_lower.endswith(".json"):
        # JSON may be either a plain list of strings or objects
        # We'll try to use LangChain's JSONLoader with a path to a key called "text" if present.
        try:
            loader = JSONLoader(file_path=temp_path, jq_schema=".[].body // .[].text // .[]")
            docs = loader.load()
            # If JSONLoader returns nothing, fallback to reading raw text fields
            if not docs:
                with open(temp_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                # convert into one big text if it's a dict
                if isinstance(j, dict):
                    text = json.dumps(j)
                    return [TextLoader(temp_path).load()[0]]
                elif isinstance(j, list):
                    # flatten to text documents
                    combined = "\n\n".join(
                        (item.get("text") or item.get("body") or json.dumps(item)) if isinstance(item, dict) else str(item)
                        for item in j
                    )
                    # create temporary file to load as text
                    return [TextLoader(temp_path).load()[0]]
            return docs
        except Exception:
            # fallback to plain text loader
            return TextLoader(temp_path).load()
    elif name_lower.endswith(".txt"):
        return TextLoader(temp_path).load()
    else:
        raise HTTPException(status_code=400, detail="Unsupported document type. Supported: pdf, json, txt.")


def build_vectorstore_from_documents(docs, embeddings) -> FAISS:
    """
    Splits documents into chunks and builds a FAISS vectorstore in-memory.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_split = text_splitter.split_documents(docs)
    vectordb = FAISS.from_documents(docs_split, embeddings)
    return vectordb


@app.post("/qa", response_class=JSONResponse)
async def question_answer(
    questions_file: UploadFile = File(..., description="JSON file containing an array of questions (strings)"),
    document_file: UploadFile = File(..., description="Document to answer against (PDF or JSON)"),
    max_answers: Optional[int] = Form(1, description="How many completions to ask the model for each question"),
):
    """
    Accepts two uploaded files:
    - questions_file: JSON array of strings (questions)
    - document_file: pdf or json file containing the document to answer from

    Returns JSON array: [{ "question": "...", "answer": "..." }, ...]
    """

    # 1) Read questions JSON
    try:
        q_bytes = await questions_file.read()
        q_text = q_bytes.decode("utf-8")
        q_json = json.loads(q_text)
        if isinstance(q_json, dict):
            # maybe {"questions": [...]} or { "items": [...] }
            # Try some sensible keys:
            for k in ("questions", "items", "qs"):
                if k in q_json and isinstance(q_json[k], list):
                    questions_list = q_json[k]
                    break
            else:
                # if dict isn't recognized, error
                raise ValueError("questions JSON must be an array of strings or contain a top-level 'questions' array")
        elif isinstance(q_json, list):
            questions_list = q_json
        else:
            raise ValueError("questions JSON must be a list")
        # validate list of strings
        questions = [str(q).strip() for q in questions_list if str(q).strip()]
        if not questions:
            raise ValueError("No questions found in uploaded JSON.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing questions file: {exc}")

    # 2) Save document to temp file and load via LangChain loader
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=document_file.filename) as tmp:
            content = await document_file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write uploaded document: {exc}")

    try:
        docs = load_document_from_file(tmp_path, document_file.filename)
        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from uploaded document.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load document: {exc}")

    # 3) Build embeddings + vectorstore
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        vectordb = build_vectorstore_from_documents(docs, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create vectorstore/embeddings: {exc}")

    # 4) Build LLM and retrieval QA chain
    try:
        # OpenAI wrapper from LangChain for LLM usage. Provide model name and max tokens if desired.
        llm = OpenAI(model_name=OPENAI_MODEL, temperature=0.0, max_tokens=1024, n=max(1, int(max_answers)))
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM or QA chain: {exc}")

    # 5) Run QA for every question (serially for simplicity). In production you might run async / batch.
    out: List[QAItem] = []
    for q in questions:
        try:
            result = qa_chain.run(q)
            out.append(QAItem(question=q, answer=str(result)))
        except Exception as exc:
            out.append(QAItem(question=q, answer=f"ERROR: failed to answer question: {exc}"))

    # cleanup (FAISS keeps memory; if needed save vectordb)
    # FAISS in-memory will be freed when process ends or vectorstore object goes out of scope.

    return JSONResponse(status_code=200, content=[item.dict() for item in out])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
