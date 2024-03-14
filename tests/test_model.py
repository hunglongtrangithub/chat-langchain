import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from custom_chat_model import CustomModel, CustomChatModel
def test_chat_model():
    model = CustomChatModel(
        "openai-community/gpt2",
        system_prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.",
    )
    very_long_text = "Hello how are you?" * 1000
    current_conversation = [{"role": "user", "content": very_long_text}]
    print(model.invoke(current_conversation))
    # for new_text in model.stream(current_conversation):
        # print(new_text, end="", flush=True)

def test_model():
    llm = CustomModel("facebook/opt-125m")
    print(llm.invoke("What is the capital of France?"*1000))
    # for new_text in llm.stream("What is the capital of France?"*1000):
        # print(new_text, end="", flush=True)


if __name__ == "__main__":
    test_chat_model()
