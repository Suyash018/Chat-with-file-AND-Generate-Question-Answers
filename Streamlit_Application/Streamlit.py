import streamlit as st
import json

from gpt import getanswer
from chat_rag import quest
from embeddings import pine_gen_embedding
from pinecone import Pinecone


import os
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
index_name = "question-maker-rag"

index= Pinecone(api_key=pinecone_api_key).Index(index_name)
index.delete(delete_all=True)

st.title("Chat with file and generate question/Answer")

def txtfile(z):
    questions_content = ""
    for question_data in z["questions"]:
        questions_content += "Question: {}\n".format(question_data["question"])

        if question_data["type"] == 'MCQs':
            questions_content += "Options:\n"
            for option in question_data["options"]:
                questions_content += "- {}\n".format(option)

        questions_content += "\nAnswer: {}\n\n".format(question_data["answer"]) 
    return questions_content

    
uploaded_files = st.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
# if not uploaded_files:
#     st.info("Please upload PDF documents to continue.")
#     st.stop()


if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False

if 'chat_start' not in st.session_state:
    st.session_state.chat_start = False

if st.button('Generate'):
    pine_gen_embedding(uploaded_files)

    st.session_state.embeddings_created = True
    st.session_state.chat_start = True



if st.session_state.embeddings_created == True:

    one_word = st.number_input('Total One word Questions',value=0,max_value =3,min_value=0,key="one_word")
    mcq = st.number_input('Total mcq Questions',value=0,max_value =3,min_value=0,key="mcq")
    True_false = st.number_input('Total True/False Questions',value=0,max_value =3,min_value=0,key="True_false")

    st.write('Total number of questions are ', one_word+True_false+mcq)

    if st.button('Start'):
        st.write("Loading")
        data = json.loads(getanswer(one_word,mcq,True_false))
        answer= txtfile(data)
        st.download_button('Download the text file and the questions', answer)
        st.write(answer)


if st.session_state.chat_start == True:


    with st.sidebar:
        messages = st.container(height=400)
        messages.chat_message("assistant").write("Hello! How can I assist you today?")
        if prompt := st.chat_input("Ask About the document"):
            messages.chat_message("user").write(prompt)
            messages.chat_message("assistant").write(quest(prompt))
