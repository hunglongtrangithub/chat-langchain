from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline

checkpoint = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
pipe = pipeline("text-generation", checkpoint)

input_text = "What is the capital of France? " * 1000
default_generation_kwargs = {
    "max_new_tokens": 512,
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": 0.1,
}
# inputs = tokenizer.encode(input_text, max_length=512, truncation=True, return_tensors="pt")
#
# # Generate text
# output_sequences = model.generate(inputs, **default_generation_kwargs)  # Adjust `max_length` as needed
#
# # Decode the generated text
# generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
generated_text = pipe(input_text, **default_generation_kwargs, max_length=1024)[0]["generated_text"]
print(generated_text)
