import torch
from transformers import AutoTokenizer
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, stops: list[str] = []):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(
        self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs
    ) -> bool:
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == self.tokenizer.decode(last_token):
                return True
        return False


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# stop = "stop"
# input = "hello stop the worldstop"
# input_ids = tokenizer(input, return_tensors="pt")["input_ids"]
# last_token = input_ids[0][-1]
# print(tokenizer.decode(last_token))

stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(tokenizer, ["stop"])])
