import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import sys
import os

@ray.remote(num_gpus=1)
class DistributedGemma7BServer:
    def __init__(self, model_name="google/gemma-7b"):
        print("=== Ray Actor Initialization ===")
        print(f"[Actor] torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"[Actor] torch.cuda.device_count(): {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"[Actor] Using device: {device}")
        else:
            device = torch.device("cpu")
            print("[Actor] WARNING: No GPU detected, using CPU.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        print(f"[Actor] Model loaded on {device}.")

    def generate(self, prompt, max_new_tokens=32):
        device = self.model.device
        print(f"[Actor] Model inference on device: {device}")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            streamer=streamer
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for chunk in streamer:
            generated_text += chunk
        thread.join()
        return generated_text

def extract_answer(result):
    answer = ""
    if "A:" in result:
        answer = result.split("A:")[1].strip().split("\n")[0]
    else:
        lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
        answer = lines[0] if lines else result.strip()
    return answer

def main():
    print("=== Ray Initialization ===")
    if not ray.is_initialized():
        ray.init()
    print(f"[Main] torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"[Main] torch.cuda.is_available(): {torch.cuda.is_available()}")

    print("=== Starting DistributedGemma7BServer Actor ===")
    server = DistributedGemma7BServer.remote()

    prompts = [
        "Q: What is the capital of Italy?\nA:",
        "Q: Who wrote 'Pride and Prejudice'?\nA:",
        "Q: What is the largest planet in our solar system?\nA:",
        "Q: What year did the Apollo 11 moon landing occur?\nA:"
    ]

    for loop in range(1, 11):
        print(f"\n=== Iteration {loop} ===")
        result_refs = [server.generate.remote(prompt) for prompt in prompts]
        results = ray.get(result_refs)
        for prompt, result in zip(prompts, results):
            answer = extract_answer(result)
            print(f"{prompt}\n{answer}\n")

if __name__ == "__main__":
    main()