from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import tempfile
import os



def pine_gen_embedding(files):


    doc = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        doc.extend(loader.load())


    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(doc)

    embeddings=OpenAIEmbeddings() # open ai emebeddings
    index_name = "question-maker-rag"

    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

