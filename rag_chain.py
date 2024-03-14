from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from custom_chat_model import CustomOpenAI, CustomChatOpenAI
from custom_chat_model import CustomModel, CustomChatModel
from custom_chat_model import Message
from jinja2 import Template

from typing import List, Optional, Dict, Tuple, Generator, Union
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., example="What's the weather like today?")
    chat_history: Optional[List[Dict[str, str]]] = None

    class Config:
        schema_extra = {
            "example": {
                "question": "Tell me more about the solar system.",
                "chat_history": [
                    {
                        "human": "hello",
                        "ai": "Hello! How can I assist you today?"
                    },
                    {
                        "human": "What's the largest planet in our solar system?",
                        "ai": "The largest planet in our solar system is Jupiter."
                    }
                ]
            }
        }



REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

RESPONSE_TEMPLATE = """\
Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

Based on the documents above, respond to the user's message: 
{question}
"""

CHAT_TEMPLATE_STRING = """{% for message in messages %}{% if message['human'] is defined %}Human: {{ message['human'] }}\n{% endif %}{% if message['ai'] is defined %}AI: {{ message['ai'] }}\n{% endif %}{% endfor %}"""

CHECKPOINT = "facebook/opt-125m"
NUM_DOCUMENTS = 6
embeddings = OpenAIEmbeddings(chunk_size=200)
vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=NUM_DOCUMENTS))
llm = CustomModel(CHECKPOINT)
# llm = CustomOpenAI()
chat_llm = CustomChatModel(CHECKPOINT)
# chat_llm = CustomChatOpenAI()

class RAGChain:
    def __init__(self, retriever, llm, chat_llm):
        self.retriever = retriever
        self.chat_llm = chat_llm
        self.llm = llm

    def format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> List[Message]:
        if not chat_history:
            return []
        formatted_chat_history = []
        for message in chat_history:
            if "human" not in message or "ai" not in message:
                raise ValueError("Each message in the chat history must have a 'human' and 'ai' key")
            human_message = {"role": "user", "content": message["human"]}
            ai_message = {"role": "assistant", "content": message["ai"]}
            formatted_chat_history.append(human_message)
            formatted_chat_history.append(ai_message)

        formatted_chat_history = [Message(**message) for message in formatted_chat_history]
        return formatted_chat_history

    def retrieve_documents(self, request: ChatRequest) -> List[Document]:
        chat_history = request.chat_history or []
        question = request.question

        if chat_history:
            template = Template(CHAT_TEMPLATE_STRING)
            serialized_chat_history = template.render(messages=chat_history)

            rephrase_question_prompt = REPHRASE_TEMPLATE.format(
                chat_history=serialized_chat_history, question=question
            )
            # print(rephrase_question_prompt)
            rephrased_question = self.llm.invoke(rephrase_question_prompt).strip()
            question = rephrased_question

        docs = retriever.get_relevant_documents(question)
        # print(docs)
        return docs

    def stream_log(self, request: ChatRequest) -> Generator[Tuple[str, str, Union[Dict, str, List]], None, None]:
        docs = self.retrieve_documents(request)

        serialized_docs = []
        for i, doc in enumerate(docs):
            doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
            serialized_docs.append(doc_string)
        serialized_docs = "\n".join(serialized_docs)
        
        user_prompt = RESPONSE_TEMPLATE.format(context=serialized_docs, question=request.question)
        formatted_chat_history = self.format_chat_history(request.chat_history)
        current_conversation = formatted_chat_history + [Message(role="user", content=user_prompt)]

        text_streamer = self.chat_llm(current_conversation, stream=True)
        formatted_docs = [doc.json() for doc in docs]

        yield "replace", "", {}
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}
        
        # NOTE: The system prompt is added by the chat llm itself. The chat history does not contain the system messages.
        response = ""
        yield "add", "/streamed_output", []
        for chunk in text_streamer:
            response += chunk
            yield "add", "/streamed_output/-", chunk
        
        yield "replace", "/final_output", {"output": response}
        
chain = RAGChain(retriever, llm, chat_llm)

if __name__ == "__main__":
    current_chat = {
        "question": "Can you give me examples of applications built with LangChain?",
        "chat_history": [
            {
                "human": "What is LangChain?",
                "ai": "LangChain is a framework designed to facilitate the building of applications that leverage large language models (LLMs) like GPT. It provides tools and abstractions to make it easier to create complex systems involving language understanding and generation.",
            },
            {
                "human": "How does LangChain work?",
                "ai": "LangChain works by providing a set of libraries and APIs that allow developers to easily integrate language models into their applications. It supports various components for dialogue management, document retrieval, and information synthesis, making it versatile for a range of use cases.",
            },
            {
                "human": "What are some key features of LangChain?",
                "ai": "Key features of LangChain include modular components for building chat systems, document retrieval and summarization tools, and integration capabilities with multiple language models and data sources. It's designed to be flexible and extensible, allowing developers to customize their applications as needed.",
            },
            {
                "human": "Can LangChain be used for building chatbots?",
                "ai": "Yes, LangChain can be used for building chatbots. It offers specialized components for handling conversations, managing context, and generating responses, making it a powerful tool for developers looking to create advanced chatbot applications.",
            },
            {
                "human": "How can LangChain integrate with other technologies?",
                "ai": "LangChain can integrate with other technologies through its flexible architecture. It supports connecting with various APIs, data sources, and language models, allowing for seamless integration with existing systems and tools. This makes it suitable for a wide range of applications beyond just language processing.",
            },
        ],
    }

    request = ChatRequest(**current_chat)

    # print(retrieve_documents(request))
    # print(rag_chain(request))
    # print(chain(request))
    for chunk in chain.stream_log(request):
        print(chunk, end="\n", flush=True)
