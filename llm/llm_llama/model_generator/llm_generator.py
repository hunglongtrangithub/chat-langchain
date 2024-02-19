from llm_pipeline import get_pipeline, stop_word_list
import argparse
import argparse
from tqdm.auto import tqdm



from negex import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument("--peft",
                        type=int,
                        default=1,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args




# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def main(args):
    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    model_id = args.path
    pipeline=get_pipeline(model_id, args.peft)
    sequences = pipeline("what is the recipe of mayonnaise?")
    for seq in sequences:
        print(seq['generated_text'])



if __name__ == "__main__":

    args = parse_args()
    main(args)