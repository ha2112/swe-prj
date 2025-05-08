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


from flask import Flask, request, jsonify



app = Flask(__name__)
load_dotenv(override=True)

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

@app.route('/process_document', methods=['POST'])
def process_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the uploaded file temporarily
    file_path = f"temp_{file.filename}"
    file.save(file_path)
    
    # Process based on file type
    if file.filename.endswith('.pdf'):
        text = pdf_text_extract(file_path)
    elif file.filename.endswith('.md'):
        text = md_text_extract(file_path)
    elif file.filename.endswith('.txt'):
        text = txt_text_extract(file_path)
    else:
        os.remove(file_path)
        return jsonify({"error": "Unsupported file format"}), 400
    
    # Clean and process the text
    cleaned_text = clean_text(text)
    condensed_text = clean_pdf_data(cleaned_text)
    
    if condensed_text[0:14] == "\n```plaintext\n":
        condensed_text = condensed_text[14:]
    
    # Create vector store
    try:
        vector_store = encode_cleaned_text(condensed_text, MONGODB_ATLAS_CLUSTER_URI)
        os.remove(file_path)
        return jsonify({
            "success": True,
            "message": "Document processed and stored successfully",
            "token_counts": {
                "raw": count_tokens_for_gemini(text),
                "cleaned": count_tokens_for_gemini(cleaned_text),
                "condensed": count_tokens_for_gemini(condensed_text)
            }
        })
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_document():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query']
    
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        DB_NAME = "langchain_test_db"
        COLLECTION_NAME = "langchain_test_vectorstores"
        ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain-test-index-vectorstores"
        
        MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
        
        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = MongoDBAtlasVectorSearch(
            collection=MONGODB_COLLECTION,
            embedding=embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )
        
        # Search for relevant documents
        results = vector_store.similarity_search(query, k=3)
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return jsonify({
            "success": True,
            "results": formatted_results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
