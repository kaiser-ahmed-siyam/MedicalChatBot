print("🔥 Starting app.py...")

from flask import Flask, jsonify, render_template, request
from src.helper import (
    download_hugging_face_embeddings,
    load_pdf_file,
    filter_to_minimal_docs,
    text_split
)
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from src.prompt import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# API Key setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY if GROQ_API_KEY else ""

# ✅ GLOBAL VARIABLES
vectorstore = None
retrieve = None
rag_chain = None


# 🔥 LAZY INITIALIZATION (FIXED)
def initialize_rag():
    global vectorstore, retrieve, rag_chain

    # ✅ Prevent reloading
    if rag_chain is not None:
        return

    print("🚀 Initializing RAG pipeline...")

    try:
        data = load_pdf_file(data='data/')
        minimal_docs = filter_to_minimal_docs(data)
        text_chunks = text_split(minimal_docs)

        vectorstore = FAISS.from_documents(
            text_chunks,
            download_hugging_face_embeddings()
        )

        retrieve = vectorstore.as_retriever()

        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
        )

        rag_chain = (
            {
                "context": retrieve,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("✅ RAG pipeline ready!")

    except Exception as e:
        print("❌ Error initializing RAG:", str(e))


# ✅ HOME ROUTE
@app.route("/")
def index():
    return render_template('templates//chat.html')


# ✅ HEALTH CHECK (IMPORTANT FOR DEBUGGING)
@app.route("/health")
def health():
    return "OK"


# ✅ CHAT ROUTE
@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        initialize_rag()   # 🔥 Lazy load here

        msg = request.form.get("msg")
        print("Question:", msg)

        if not msg:
            return "No input provided"

        response = rag_chain.invoke(msg)

        print("Response:", response)

        return str(response)

    except Exception as e:
        print("❌ Error in chat:", str(e))
        return "Error processing request"


# ✅ RUN APP (FOR LOCAL ONLY)
if __name__ == '__main__':
    print("🔥 Starting Flask app...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)


# print("🔥 Starting app.py...")
# from flask import Flask, jsonify, render_template, request
# from src.helper import (
#     download_hugging_face_embeddings,
#     load_pdf_file,
#     filter_to_minimal_docs,
#     text_split
# )
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import os

# app = Flask(__name__)

# load_dotenv()

# # ✅ API Key
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# # ✅ GLOBAL VARIABLES (initially empty)
# vectorstore = None
# retrieve = None
# rag_chain = None


# # 🔥 LAZY INITIALIZATION (VERY IMPORTANT)
# def initialize_rag():
#     global vectorstore, retrieve, rag_chain

#     if rag_chain is None:
#         print("🚀 Initializing RAG pipeline...")

#         data = load_pdf_file(data='data/')
#         minimal_docs = filter_to_minimal_docs(data)
#         text_chunks = text_split(minimal_docs)

#         vectorstore = FAISS.from_documents(
#             text_chunks,
#             download_hugging_face_embeddings()
#         )

#         retrieve = vectorstore.as_retriever()

#         llm = ChatGroq(
#             model="openai/gpt-oss-20b",
#             temperature=0,
#         )

#         rag_chain = (
#             {
#                 "context": retrieve,
#                 "question": RunnablePassthrough()
#             }
#             | prompt
#             | llm
#             | StrOutputParser()
#         )

#         print("✅ RAG pipeline ready!")


# @app.route("/")
# def index():
#     return render_template('chat.html')

# initialize_rag()  # 🔥 Load only when needed

# @app.route("/get", methods=["GET","POST"])
# def chat():
    

#     msg = request.form.get("msg")
#     print("Question:", msg)

#     response = rag_chain.invoke(msg)

#     print("Response:", response)
#     # return jsonify({"answer": str(response)})
#     return str(response)


# # ✅ REQUIRED FOR RENDER
# if __name__ == '__main__':
#     print("🔥 Starting Flask app...")
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port, debug=True)
#     # app.run(host="0.0.0.0", port=1441, debug=True)

# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings, load_pdf_file, filter_to_minimal_docs, text_split
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *

# from langchain_core.documents import Document
# import os

# app = Flask(__name__)

# load_dotenv()

# GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

# os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# # Load your PDF documents here (replace with your actual document loading logic)
# from langchain_community.document_loaders import PyPDFLoader
# data = load_pdf_file(data = 'data/')
# minimal_docs = filter_to_minimal_docs(data)
# text_chunks = text_split(minimal_docs)


# vectorsstore = FAISS.from_documents(text_chunks,
#                                      download_hugging_face_embeddings())

# # retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
# retrieve = vectorsstore.as_retriever()

# llm=ChatGroq(
#     model="openai/gpt-oss-20b",
#     temperature=0,
    
# )

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # retriever = your_vectorstore.as_retriever()

# rag_chain = (
#     {
#         "context": retrieve,   # retrieves relevant chunks
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     question = msg
#     print(question)
#     response = rag_chain.invoke(msg)
#     print("Response : ", response)
#     return str(response)



# if __name__ == '__main__':
#      port = int(os.environ.get("PORT", 10000))
#      app.run(host="0.0.0.0", port=port, debug=True)
    