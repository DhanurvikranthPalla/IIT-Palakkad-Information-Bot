from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import os
os.environ["OPENAI_API_KEY"] = ""    # insert API key here

pdfreader = PdfReader('IIT_Palakkad_Data.pdf')   # Loading the data

from typing_extensions import Concatenate

# Enumerating the data
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Spliting the the data into chunks
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Creating embeddings and feeding the data
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")   # Loading the chain function

# Creating a loop for bot
while True:
    query = input("Ask a question: ")
    if query.lower() == 'exit':
        break    
    else:    
        docs = document_search.similarity_search(query)
        chain.run(input_documents=docs, question=query)