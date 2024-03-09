import asyncio
from ..chain import answer_chain
from langchain.schema.runnable import RunnableLambda

# from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# checkpoint = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)
# streamer = TextStreamer(tokenizer)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, streamer=streamer)
# llm = HuggingFacePipeline(pipe)

input = {
    "question": "What is the capital of Vietnam?",
    "chat_history": [
        {
            "human": "What is the capital of France?",
            "ai": "The capital of France is Paris.",
        },
    ],
}


def test_invoke():
    print(answer_chain.invoke(input))


def test_astream_log():
    async def log_messages(response):
        async for message in response:
            print(message.ops[0]["value"])

    asyncio.run(log_messages(answer_chain.astream_log(input)))
