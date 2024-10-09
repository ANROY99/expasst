from flask import Flask, request
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
import os
import config

app = Flask(__name__)

@app.route('/result', methods=["GET", "POST"])
def result():

    DB_FAISS_PATH = os.getcwd()

    pdf_loader = PyPDFLoader("Sample_Travelling_Allowances_Sanit.pdf")

    output = request.get_json()
    user_query = output['userquery']

    loaders = [pdf_loader]

    documents = []
    for loader in loaders:
      documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    all_documents = text_splitter.split_documents(documents)

    print(f"Total number of documents: {len(all_documents)}")

    co = cohere.Client(config.api_key)

    cohere_embeddings = CohereEmbeddings(cohere_api_key=config.api_key,user_agent='langchain.partner')

    db = FAISS.from_documents(documents, cohere_embeddings)
    db.save_local(folder_path=DB_FAISS_PATH)
    db = FAISS.load_local(DB_FAISS_PATH, cohere_embeddings, allow_dangerous_deserialization=True)

    retv = db.as_retriever(search_kwargs={"k": 3})

    docs = retv.invoke(user_query)


    selected_doc=""
    for doc in docs:
      selected_doc=selected_doc + "\n" + doc.page_content

    selected_doc = "Following is a portion of the expense reimbersement policy of the company. " + selected_doc

    selected_doc = selected_doc + " \n Based on the above document, provide a short answer the the following query :"+ user_query

    response = co.chat(
           model='command-r-plus',
           message=selected_doc,
           temperature=1,
           prompt_truncation='AUTO'
    )

    print(response.text)


    cohere_res = {}
    cohere_res['answer'] = response.text

    return (cohere_res)

if __name__ == '__main__':
    app.run()

