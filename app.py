import streamlit as st 
from llama_index import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex
from llama_index.llms import OpenAI 
import os
import openai
from dotenv import load_dotenv
load_dotenv()

# Setting an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.header("Chat with your data")

    with st.sidebar:
        st.title('Chat with your data')
        st.markdown('''
        ## About 
        This app is an LLM powered chatbot built using 
        - [Streamlit](https://streamlit.io)
        - [LLAMA Index](https://gpt-index.readthedocs.io)
        - [Open AI](https://platform.openai.com/docs/models) LLM Model
        ''')

    reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
    docs = reader.load_data()

    service_context = ServiceContext.from_defaults(llm=OpenAI('gpt-3.5-turbo', temperature=0.5), system_prompt='you are a machine learning engineer and your job is to technical questions')

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    # take query and relative documents and give those to language model 
    query = st.text_input("Ask questions related to your data")
    if query:
        chat_engine = index.as_chat_engine(chat_mode='condense_question', verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)


if __name__ == "__main__":
    main()