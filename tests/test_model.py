from ..custom_chat_model import CustomChatModel, CustomModel
import sys


def test_chat_model(stream=False):
    model = CustomChatModel(
        "facebook/opt-125m",
        system_prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.",
    )
    print("System prompt: " + model.chat_history[0]["content"])
    print("=" * 40)
    while True:
        print(len(model.chat_history))
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("AI: ", end="")
        if stream:
            for new_text in model.stream(user_input):
                print(new_text, end="", flush=True)
        else:
            print(model.invoke(user_input))
        print()
        print("=" * 40)


def test_model():
    llm = CustomModel("facebook/opt-125m")
    print(llm.invoke("What is the capital of France?"))
    for new_text in llm.stream("What is the capital of France?"):
        print(new_text, end="", flush=True)


if __name__ == "__main__":
    test_chat_model()
