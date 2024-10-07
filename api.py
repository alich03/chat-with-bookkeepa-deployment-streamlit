from fastapi import FastAPI
from pydantic import BaseModel

from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import json
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

class ChatwithDoc(BaseModel):
    prompt: str
    
@app.post("/bookkeepa-rag/")
def rag(data: ChatwithDoc):
    user_prompt = data.prompt

    llm = ChatOpenAI(
    api_key=os.environ.get('OPEN_AI_KEY'),
    model='chatgpt-4o-latest',  # or 'gpt-3.5-turbo'
    temperature=0.3,
    max_tokens=5096 )
    embed_model = OpenAIEmbeddings(api_key=os.getenv("OPEN_AI_KEY"))

    index_name = "rag1" 

    pcv= PineconeVectorStore(index_name=index_name,embedding=embed_model,pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    retriever = pcv.as_retriever(search_type="similarity", search_kwargs={"k": 7})

    retrieval_qa_chat_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=
            '''
            %INSTRUCTIONS:
            You are Doctor Bookkeepa, a chatbot designed to assist users based on a given manual or FAQ document. Your task is to guide user with relevant context from the document based on the user's prompt and provide accurate instructions without adding or omitting steps. If the context contains image urls, please extract and include those as well. Please ensure that we need a proper list of URLs. Ensure that your responses are concise, precise, and strictly aligned with the manual's content.

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
    response = retrieval_chain.invoke({"input":user_prompt})
    structured_answer = response['answer']

    try:
        structured_answer = json.loads(structured_answer)
    except:
        pass
    # image_links = structured_answer['image_urls'] 

    return structured_answer
