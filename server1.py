from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod

from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
from pprint import pprint
import os
from datasets import Dataset
from typing_extensions import TypedDict
from IPython.display import display, Image
from typing import List, TypedDict

from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity
)

import langgraph
from pymongo import MongoClient

from my_helper_function import md_text_extract, pdf_text_extract, txt_text_extract,\
clean_pdf_data, clean_text, count_tokens_for_gemini, cluster_chunking


load_dotenv(override=True)


pdf_path = "Computer_Network_Chapter_3.pdf"
pdf_text = pdf_text_extract(pdf_path)
print(pdf_text)


md_path = "Computer_Network_Chapter_3.md"
md_text = md_text_extract(md_path)
print(md_text)


txt_path = "Computer_Network_Chapter_3.txt"
txt_text = txt_text_extract(txt_path)
print(txt_text)


cleaned_pdf_text = clean_text(pdf_text)
cleaned_txt_text = clean_text(txt_text)
cleaned_md_text = clean_text(md_text)



condensed_pdf_text = clean_pdf_data(cleaned_pdf_text)

if(condensed_pdf_text[0:14] == "\n```plaintext\n"):
    condensed_pdf_text = condensed_pdf_text[14:]


num_token_raw = count_tokens_for_gemini(pdf_text)
num_token_cleaned = count_tokens_for_gemini(cleaned_pdf_text)
num_token_condensed = count_tokens_for_gemini(condensed_pdf_text)
print(f"""
    {num_token_raw}
    ------
    {num_token_cleaned}
    ------
    {num_token_condensed}
""")

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

def encode_cleaned_text(str, mongo_db_uri):
    client = MongoClient(mongo_db_uri)  

    DB_NAME = "langchain_test_db"
    COLLECTION_NAME = "langchain_test_vectorstores"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain-test-index-vectorstores"

    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    MONGODB_COLLECTION.delete_many({})

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = MongoDBAtlasVectorSearch(
        collection=MONGODB_COLLECTION,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )
    vector_store.create_vector_search_index(dimensions=768)
    

    chunks = cluster_chunking(str)
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={"source": "pdf"} 
        )
        documents.append(doc)


    vector_store.add_documents(documents)
    return vector_store


chunks_vector_store = encode_cleaned_text(str = condensed_pdf_text, mongo_db_uri= MONGODB_ATLAS_CLUSTER_URI)