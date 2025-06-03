import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import sys  # For printing updates on same line

@ray.remote
class DistributedGemma7BServer:
    def __init__(self, model_name="google/gemma-7b"):
        print("Loading distributed Gemma-7B...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded in distributed mode.")

    def generate(self, prompt, max_new_tokens=64):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            streamer=streamer
        )

        # Launch generation in a thread so we can iterate over tokens
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        token_count = 0
        for chunk in streamer:
            generated_text += chunk
            token_count = len(self.tokenizer(generated_text, return_tensors="pt").input_ids[0])
            print(f"\rLive token count: {token_count}", end="", file=sys.stderr)  # Print on same line to stderr
        thread.join()
        print("\n", file=sys.stderr)  # Move to next line after generation

        return prompt + generated_text

def main():
    print(f"CUDA GPUs available: {torch.cuda.device_count()}")
    server = DistributedGemma7BServer.remote()
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function for quicksort.",
        "Describe why the sky is blue."
    ]
    results = ray.get([server.generate.remote(p) for p in prompts])
    for i, res in enumerate(results):
        print(f"Prompt: {prompts[i]}\n---\n{res}\n{'='*40}")

if __name__ == "__main__":
    main()