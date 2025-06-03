import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import sys

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

def main():
    print("=== Ray Initialization ===")
    if not ray.is_initialized():
        ray.init()
    print(f"[Main] torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"[Main] torch.cuda.is_available(): {torch.cuda.is_available()}")

    print("=== Starting DistributedGemma7BServer Actor ===")
    server = DistributedGemma7BServer.remote()

    prompt = "What is the capital of Italy?"
    print(f"=== Sending prompt to actor: '{prompt}' ===")

    result = ray.get(server.generate.remote(prompt))
    print(f"=== Model Output ===\n{result}")

if __name__ == "__main__":
    main()