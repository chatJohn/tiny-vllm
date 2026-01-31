from transfomers import AutoTokenizer
import sys
import os
from src import LLM, SamplingParams
def main():
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    model_name = "Qwen3-0.6B"
    model_path = os.path.join(DIR_PATH, f"models/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast= True)
    llm = LLM(model_path)
    sampling_params = SamplingParams()
    prompts = [
        "Who you are?",
        "Define yourself in one sentence.",
    ]
    prompts = [
        tokenizer.encode(
            [{"role": user, "content": prompt}],
            tokenize = False, # don't tokenize yet
            add_generation_token = True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompts: {prompt!r}")
        print(f"Outputs: {output["text"]!r}")
    

if __name__ == "__main__":
    main()