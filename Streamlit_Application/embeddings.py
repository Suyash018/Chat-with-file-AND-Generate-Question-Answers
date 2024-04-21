from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import tempfile
import os



def pine_gen_embedding(files):

    from pinecone import Pinecone
    import os
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    index_name = "question-maker-rag"
    index= Pinecone(api_key=pinecone_api_key).Index(index_name)
    index.delete(delete_all=True)




    doc = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
            print(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        doc.extend(loader.load())


    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(doc)
    print("\n\n\n\ndocs\n\n",docs )

    embeddings=OpenAIEmbeddings() # open ai emebeddings
    index_name = "question-maker-rag"

    docsss=PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = docsss.as_retriever()


