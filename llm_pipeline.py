from torch import cuda, bfloat16
import transformers
import torch
from huggingface_hub import login

login()

DEVICE = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def get_pipeline(model_id):
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = (
        transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )
        if torch.cuda.is_available()
        else None
    )
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # enable evaluation mode to allow model inference
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    text_generator = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task="text-generation",
    )

    return text_generator


if __name__ == "__main__":
    MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
    from langchain.llms import HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=get_pipeline(MODEL_ID))
    response = llm.invoke("Hello")
    print(response)
