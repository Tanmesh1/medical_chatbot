from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

from src.helper import download_embeddings
from src.prompt import MEDICAL_RAG_PROMPT

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()

app = Flask(__name__)

#----------------------
# Connect to local LLM
#----------------------

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0.3,
)

#----------------------------
# Local Embeddings + Pinecone Index
#----------------------------

embeddings = download_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

#---------------------------
# CREATE MEDICAL RAG PROMPT
#---------------------------

prompt = ChatPromptTemplate.from_template(MEDICAL_RAG_PROMPT)

#---------------------------
# CREATE RAG CHAIN
#---------------------------

qa_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,qa_chain)

#--------------------------
# Routes
#--------------------------

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat",methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = rag_chain.invoke({"input": user_message})

    answer = response["answer"]

    return jsonify({"reply": answer})

#-------------------
# Run app
#-------------------

if __name__ =='__main__':
    app.run(host="0.0.0.0", port =8080 ,debug = True)