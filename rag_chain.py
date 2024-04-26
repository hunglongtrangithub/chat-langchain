from math import log
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.document import Document

from jinja2 import Template
from typing import List, Optional, Dict, Tuple, Union, Generator, AsyncGenerator
from pydantic import BaseModel, Field

from logger_config import get_logger
from custom_chat_model import BaseChat, chat_llm
from vectorstore import vectorstore
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    filename="chat_history.log",
)

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(..., examples=["What's the weather like today?"])
    chat_history: Optional[List[Dict[str, str]]] = None

    class Config:
        schema_extra = {
            "example": {
                "question": "Tell me more about the solar system.",
                "chat_history": [
                    {"human": "hello", "ai": "Hello! How can I assist you today?"},
                    {
                        "human": "What's the largest planet in our solar system?",
                        "ai": "The largest planet in our solar system is Jupiter.",
                    },
                ],
            }
        }


def log_chat_history(docs: List[Document], request: ChatRequest, response: str):
    logging.info("=" * 80)
    logging.info(f"Received request: {request.model_dump_json()}")
    logging.info(
        f"Retrieved {len(docs)} documents.\n"
        + "\n".join(
            [f"<doc id='{i}'>{doc.page_content}</doc>" for i, doc in enumerate(docs)]
        )
    )
    logging.info(f"Generated response: {response}")


# System prompt for Langchain chatbot
SYSTEM_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""
# System prompt for medical chatbot
SYSTEM_TEMPLATE = """\
You are an excellent medical assistant, tasked with answering any question about \
medical information from the provided search results.

Always answer as truthfully as possible, to the best of your knowledge and the \
information provided in the search results.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""
REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

# Jinja2 template for formatting chat history
CHAT_TEMPLATE_STRING = """\
{%- for message in messages %}
{% if message.role == 'user' -%}
Human: {{ message.content -}}\n
{%- elif message.role == 'assistant' -%}
AI: {{ message.content -}}\n
{%- else -%}
Unknown Role: {{ message.content -}}\n
{%- endif -%}
{%- endfor -%}\
"""

NUM_DOCUMENTS = 3
retriever = vectorstore.as_retriever(search_kwargs=dict(k=NUM_DOCUMENTS))


class RAGChain:
    def __init__(self, retriever: VectorStoreRetriever, chat_llm: BaseChat):
        self.retriever = retriever
        self.chat_llm = chat_llm

    def format_chat_history(
        self, chat_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        if not chat_history:
            return []
        formatted_chat_history = []
        for message in chat_history:
            if "human" not in message or "ai" not in message:
                raise ValueError(
                    "Each message in the chat history must have a 'human' and 'ai' key"
                )
            human_message = {"role": "user", "content": message["human"]}
            ai_message = {"role": "assistant", "content": message["ai"]}
            formatted_chat_history.append(human_message)
            formatted_chat_history.append(ai_message)

        return formatted_chat_history

    async def aretrieve_documents(self, request: ChatRequest) -> List[Document]:
        chat_history = self.format_chat_history(request.chat_history)
        question = request.question

        if chat_history:
            try:
                template = Template(CHAT_TEMPLATE_STRING)
                serialized_chat_history = template.render(messages=chat_history).strip()
            except Exception as e:
                logger.error("Error occurred while applying template to chat history.")
                serialized_chat_history = ""

            rephrase_question_prompt = REPHRASE_TEMPLATE.format(
                chat_history=serialized_chat_history, question=question
            )
            try:
                rephrased_question = await self.chat_llm.ainvoke(
                    [{"role": "user", "content": rephrase_question_prompt}],
                )
                question = rephrased_question.strip()
            except Exception as e:
                logger.error(
                    "Error occurred while invoking chat model to rephrase question."
                )
        try:
            docs = await self.retriever.aget_relevant_documents(question)
        except Exception as e:
            logger.error("Error occurred while retrieving documents:", e)
            docs = []
        return docs

    def retrieve_documents(self, request: ChatRequest) -> List[Document]:
        chat_history = self.format_chat_history(request.chat_history)
        question = request.question

        if chat_history:
            try:
                template = Template(CHAT_TEMPLATE_STRING)
                serialized_chat_history = template.render(messages=chat_history).strip()
            except Exception as e:
                logger.error("Error occurred while applying template to chat history")
                serialized_chat_history = ""

            rephrase_question_prompt = REPHRASE_TEMPLATE.format(
                chat_history=serialized_chat_history, question=question
            )
            try:
                rephrased_question = self.chat_llm.invoke(
                    [{"role": "user", "content": rephrase_question_prompt}],
                )
                question = rephrased_question.strip()
            except Exception as e:
                logger.error(
                    "Error occurred while invoking chat model to rephrase question",
                )
        try:
            docs = self.retriever.get_relevant_documents(question)
        except Exception as e:
            logger.error("Error occurred while retrieving documents:", e)
            docs = []
        return docs

    def get_response_streamer_with_docs(
        self, request: ChatRequest, docs: List[Document]
    ) -> Generator[str, None, None]:
        serialized_docs = "\n".join(
            [f"<doc id='{i}'>{doc.page_content}</doc>" for i, doc in enumerate(docs)]
        )

        system_prompt = SYSTEM_TEMPLATE.format(context=serialized_docs)
        formatted_chat_history = self.format_chat_history(request.chat_history)
        current_conversation = [
            {"role": "system", "content": system_prompt},
            *formatted_chat_history,
            {"role": "user", "content": request.question},
        ]
        try:
            text_streamer = self.chat_llm.stream(current_conversation)
            return text_streamer  # type: ignore
        except Exception as e:
            logger.error("Error occurred while streaming response")
            yield ""

    def aget_response_streamer_with_docs(
        self, request: ChatRequest, docs: List[Document]
    ) -> AsyncGenerator[str, None]:
        serialized_docs = "\n".join(
            [f"<doc id='{i}'>{doc.page_content}</doc>" for i, doc in enumerate(docs)]
        )

        system_prompt = SYSTEM_TEMPLATE.format(context=serialized_docs)
        formatted_chat_history = self.format_chat_history(request.chat_history)
        current_conversation = [
            {"role": "system", "content": system_prompt},
            *formatted_chat_history,
            {"role": "user", "content": request.question},
        ]
        try:
            text_streamer = self.chat_llm.astream(current_conversation)
            return text_streamer
        except Exception as e:
            logger.error("Error occurred while streaming response")

            async def empty_streamer():
                yield ""

            return empty_streamer()

    async def astream_log(
        self, request: ChatRequest
    ) -> AsyncGenerator[Tuple[str, str, Union[Dict, str, List]], None]:
        yield "replace", "", {}

        docs = await self.aretrieve_documents(request)
        formatted_docs = [doc.json() for doc in docs]
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}

        text_streamer = self.aget_response_streamer_with_docs(request, docs)
        response = ""
        yield "add", "/streamed_output", []
        async for chunk in text_streamer:
            response += chunk
            yield "add", "/streamed_output/-", chunk
        log_chat_history(docs, request, response)
        yield "replace", "/final_output", {"output": response}

    def stream_log(
        self, request: ChatRequest
    ) -> Generator[Tuple[str, str, Union[Dict, str, List]], None, None]:
        docs = self.retrieve_documents(request)
        formatted_docs = [doc.json() for doc in docs]
        text_streamer = self.get_response_streamer_with_docs(request, docs)

        yield "replace", "", {}
        yield "add", "/logs", {}
        yield "add", "/logs/FindDocs", {}
        yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}

        response = ""
        yield "add", "/streamed_output", []
        for chunk in text_streamer:
            response += chunk
            yield "add", "/streamed_output/-", chunk
        log_chat_history(docs, request, response)
        yield "replace", "/final_output", {"output": response}

    async def ainvoke_log(self, request: ChatRequest) -> Dict[str, Union[str, List]]:
        docs = await self.aretrieve_documents(request)
        text_streamer = self.aget_response_streamer_with_docs(request, docs)
        response = ""
        async for chunk in text_streamer:
            response += chunk
        log_chat_history(docs, request, response)
        return {"response": response, "docs": [doc.json() for doc in docs]}


chain = RAGChain(retriever, chat_llm)
