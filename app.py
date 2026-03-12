from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf_file, filter_to_minimal_docs, text_split
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *

from langchain_core.documents import Document
import os

app = Flask(__name__)

load_dotenv()

GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

# index_name = "medical-bot" 
# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# Load your PDF documents here (replace with your actual document loading logic)
from langchain_community.document_loaders import PyPDFLoader
data = load_pdf_file(data='D:\\GanAi projects\\MedicalChatBot\\data')

pdf_gen=""
for page in data:
    pdf_gen += page.page_content


document_pdf_gen = [Document(page_content= t) for t in text_split(pdf_gen)]

vectorsstore = FAISS.from_documents(document_pdf_gen,
                                     download_hugging_face_embeddings())

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
retrieve = vectorsstore.as_retriever()

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    
)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# retriever = your_vectorstore.as_retriever()

rag_chain = (
    {
        "context": retrieve,   # retrieves relevant chunks
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    question = msg
    print(question)
    response = rag_chain.invoke(msg)
    print("Response : ", response)
    return str(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
