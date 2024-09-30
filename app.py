import streamlit as st
import requests
import json
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
load_dotenv()
from time import sleep

# os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
# os.environ['openai_api_key'] = os.getenv("OPEN_AI_KEY")


def chatbot_rag():
    st.title("Chat with PDFs BOOKKEEPA")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        with st.chat_message("assistant"):
            st.markdown("How can I help you?")
            
    # Display all messages stored in session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content_text"])
            if message["content_urls"]:
                image_links = message["content_urls"]
                if isinstance(image_links, list):
                    # Display multiple images side by side using columns
                    cols = st.columns(len(image_links))
                    for col, image_link in zip(cols, image_links):
                        col.image(image_link, caption='Image from manual', width=100, use_column_width=False)
                else:
                    # Display single image
                    st.image(image_links, caption='Image from manual', width=100, use_column_width=False)

    # Input from user
    if prompt := st.chat_input("How can I help you for Bookkeepa ..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content_text": prompt, "content_urls": None})

        # Get response from API
        response_text = get_response_from_api(prompt)

        structured_answer = json.loads(response_text)
        image_links = structured_answer['image_urls']  # Assume this is a list of image URLs
        
        # Display bot response and images
        with st.chat_message('assistant'):
            # st.markdown(structured_answer['response'])
            # st.markdown(image_links)
            progress_text = st.empty()

            for i in range(len(structured_answer['response'])):
                progress_text.markdown(structured_answer['response'][:i + 1])
                sleep(0.0001)

            if image_links:
                if isinstance(image_links, list):
                    # Display multiple images side by side
                    images_per_row = 5
# Iterate over the image links in chunks of 5
                    for i in range(0, len(image_links), images_per_row):
                        # Create a row of 5 columns (or fewer for the last row)
                        cols = st.columns(min(images_per_row, len(image_links) - i))
                        # Render each image in its corresponding column
                        for j, img_url in enumerate(image_links[i:i + images_per_row]):
                            with cols[j]:
                                st.image(img_url, width=100, use_column_width=False)  # Adjust image width
                else:
                    st.image(image_links, width=100, use_column_width=False)

        # Store bot response in session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content_text": structured_answer['response'], 
            "content_urls": image_links
        })





# API call

def get_response_from_api(user_input):
    llm = ChatOpenAI(
    api_key=os.environ.get('OPEN_AI_KEY'),
    model='gpt-4',  # or 'gpt-3.5-turbo'
    temperature=0.4,
    max_tokens=4096  # Adjust token limit as per the model
        )
    embed_model = OpenAIEmbeddings(api_key=os.getenv("OPEN_AI_KEY"))

    index_name = "rag1" 

    pcv= PineconeVectorStore(index_name=index_name,embedding=embed_model,pinecone_api_key=os.getenv("PINECONE_API_KEY"))

    # vector_store = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embed_model)
    retriever = pcv.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    retrieval_qa_chat_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template='''
            %INSTRUCTIONS:
            You are Doctor Bookkeepa, a chatbot designed to assist users with our app. Your main task is to retrieve relevant context based on the user’s prompt and provide appropriate instructions. If the context contains image urls, please extract and include those as well and please dont cut them off, give the entire urls. 

            Your response must strictly adhere to the following JSON format:

            {{
                "response": "<ACTUAL RESPONSE HERE>",
                "image_urls": "<IMAGE URLS LIST HERE>",

            }}
            %CONTEXT: {context}
            %QUESTION: {input}

            ''')


    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        # response = retrieval_chain.invoke({"input": "how to Transferring funds? "})
    response = retrieval_chain.invoke({"input":user_input})
    structured_answer = response['answer']

    return structured_answer





if __name__ == "__main__":
    chatbot_rag()
