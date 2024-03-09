from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List, Dict, Optional
from pydantic import BaseModel
from custom_chat_model import CustomModel, CustomChatModel
from jinja2 import Template


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

RESPONSE_TEMPLATE = """\
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
TEMPLATE_STRING = """{% for message in messages %}{% if message['human'] is defined %}Human: {{ message['human'] }}\n{% endif %}{% if message['ai'] is defined %}AI: {{ message['ai'] }}\n{% endif %}{% endfor %}"""


embeddings = OpenAIEmbeddings(chunk_size=200)
vectorstore = FAISS.load_local(
    "faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=6))
llm = CustomModel("meta-llama/Llama-2-7b-hf")
chat_llm = CustomChatModel("meta-llama/Llama-2-7b-hf")


def retrieve_documents(request: ChatRequest):
    chat_history = request.chat_history or []
    question = request.question

    if chat_history:
        template = Template(TEMPLATE_STRING)
        serialized_chat_history = template.render(messages=chat_history)

        rephrase_question_prompt = REPHRASE_TEMPLATE.format(
            chat_history=serialized_chat_history, question=question
        )

        rephrased_question = llm.invoke(rephrase_question_prompt).strip()
        request.question = rephrased_question

    docs = retriever.get_relevant_documents(request.question)

    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


if __name__ == "__main__":
    current_chat = {
        "question": "What is the capital of Vietnam?",
        "chat_history": [
            {
                "human": "What is the capital of France?",
                "ai": "The capital of France is Paris.",
            },
            {
                "human": "What is the capital of Japan?",
                "ai": "The capital of Japan is Tokyo.",
            },
            {
                "human": "What is the capital of South Korea?",
                "ai": "The capital of South Korea is Seoul.",
            },
        ],
    }

    request = ChatRequest(**current_chat)

    print(retrieve_documents(request))
