import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import openai
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain():
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory
    )
    return conversation_chain


def handle_userinput(replies, queries):
    # response = st.session_state.conversation({'question': user_question})
    # st.session_state.chat_history = response['chat_history']
    print(replies,queries)

    for i,j in zip(queries, replies):
        
        st.write(user_template.replace(
            "{{MSG}}", i), unsafe_allow_html=True)
        
        st.write(bot_template.replace(
                "{{MSG}}", j), unsafe_allow_html=True)


def main():
    # openai.api_key = "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"
    openai.api_key=st.secrets["OPENAI_KEY"]
    queries=[]
    conversation=""
    load_dotenv()
    messages = [
        {"role": "system", "content": "You are a Medical assistant . Your tone is of a therapist calming and reassuring "},
    ]
    replies=[]
    if "conversation" not in st.session_state:
        st.session_state.conversation = ""
    if "replies" not in st.session_state:
        st.session_state.replies = []
    if "queries" not in st.session_state:
        st.session_state.queries = []
    replies=st.session_state['replies']
    queries=st.session_state['queries']
    conversation=st.session_state['conversation']
    messages.append({"role": "user", "content": conversation})
    st.set_page_config(page_title="Kaira asks",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

   
    
    st.header("kaira talks")
    user_question = st.text_input("Ask a question?")
    user_question=conversation+user_question
    if user_question:
        messages = [
        {"role": "system", "content": ""},
    ]
  
    
    
    
        # message = df['Question'][0]
        message=user_question
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )

        reply = chat.choices[0].message.content
        replies.append(reply)
        conversation=conversation+reply
        queries.append(user_question)
        # print(f"ChatGPT: {reply}")
        
        
        handle_userinput(replies,queries)
        st.session_state['conversation'] = conversation
        st.session_state['replies'] = replies
        st.session_state['queries']=queries
    # st.session_state.conversation = get_conversation_chain(
                # )
    # with st.sidebar:
    #     st.subheader("Your documents")
    #     pdf_docs = st.file_uploader(
    #         "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    #     if st.button("Process"):
    #         with st.spinner("Processing"):
    #             # get pdf text
    #             raw_text = get_pdf_text(pdf_docs)

    #             # get the text chunks
    #             text_chunks = get_text_chunks(raw_text)

    #             # create vector store
    #             vectorstore = get_vectorstore(text_chunks)

    #             # create conversation chain
                # st.session_state.conversation = get_conversation_chain(
                #     vectorstore)


if __name__ == '__main__':
    main()