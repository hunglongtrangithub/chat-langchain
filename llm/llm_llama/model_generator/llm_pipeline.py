import torch
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    LlamaForCausalLM,
)
from peft import PeftModel, PeftConfig
import torch
import torch.nn as nn
import os

# stop_word_list = ["Human:"]
# max_new_tokens = 512
# max_length_stop_search = 8


# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float16)


def load_fine_tuned_model(path, peft_model):
    model_id = path
    config = PeftConfig.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        config.base_model_name_or_path, return_dict=True, device_map="auto"
    )

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.paddings_side = "left"
    print("max length", tokenizer.model_max_length)
    print(peft_model)
    if peft_model == 1:
        peft_model = PeftModel.from_pretrained(model, model_id)
    else:
        peft_model = model

    return peft_model, tokenizer


def get_stopping_criteria(tokenizer):

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            input_token_ids = (
                input_ids[0][-max_new_tokens:]
                if len(input_ids[0]) > max_new_tokens
                else input_ids[0]
            )  # cap input tokens to the last max_new_tokens tokens to ensure that the substring "AI:" will always be present
            input_text = tokenizer.decode(input_token_ids)
            # input_text will be guaranteed to have the substring "AI:". Now we find the position of the most recent "AI:" substring
            AI_response_position = input_text.rfind("AI:")
            AI_response = input_text[
                AI_response_position + len("AI:") :
            ]  # We have successfully extracted the most recent AI response from input_ids
            flag = False
            for stop_word in stop_word_list:
                if AI_response.find(stop_word) != -1:
                    flag = True
                    break
            # print(AI_response, flag)
            # print(tokenizer.decode(input_ids[0]))
            return flag

    # This method works because: at the beginning of every __call__ method being called, there will always be an "AI:" prefix at the end of the conversation history (which is included in input_ids in token id form). So the logic above will always be able to look for it, and be successful at taking only the response from the AI to check with stop words. No human input can interfere with the check.
    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    return stopping_criteria


def get_pipeline(model_id, peft_model):
    print(model_id)
    # model_config = AutoConfig.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True,config=model_config,device_map='auto')
    # # enable evaluation mode to allow model inference
    # # model.eval()

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, tokenizer = load_fine_tuned_model(model_id, peft_model)

    generator = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task="text-generation",
        # we pass model parameters here too
        # stopping_criteria=get_stopping_criteria(tokenizer),  # without this model rambles during chat
        # max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # without this output begins repeating
    )

    return generator
