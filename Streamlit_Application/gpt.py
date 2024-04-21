from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore


# Load data
embeddings=OpenAIEmbeddings() # open ai emebeddings
index_name = "question-maker-rag"
db = PineconeVectorStore(index_name=index_name, embedding=embeddings)# loading saved in pinecone

# retriver added 
retriever = db.as_retriever()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)



# Prompt
template = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
Given a piece of context, you must come up with a question and answer pair that can be used to test a student's reading comprehension abilities.
When coming up with this question/answer pair

Please come up with a question/answer pair, in JSON format, for the following context:
----------------
{context}

The User will specify how many and what type of questions it wants by {question}

The type of questions can be of  three categories: 
1.True or False 
2.Multiple Choice Questions (MCQs)
3.one-word answers.

Specify the type of each question as
1.True or False = True/False
2.Multiple Choice Questions (MCQs) = MCQs
3.one-word answers. = one-word answer

"""
custom_rag_prompt = PromptTemplate.from_template(template)


rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

def getanswer(a,b,c):
    mcq=a
    OneWord =b
    T_F=c

    Total = mcq+OneWord+T_F
    z = f"Total {Total} questions, {T_F} of which are True or False questions, {mcq} are Multiple Choice Questions (MCQs), and {OneWord} are one-word answer questions."

    ans=rag_chain.invoke(z)
    return ans
