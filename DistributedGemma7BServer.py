import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

@ray.remote
class DistributedGemma7BServer:
    def __init__(self, model_name="google/gemma-7b"):
        print("Loading distributed Gemma-7B...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # Hugging Face will shard across GPUs
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Model loaded in distributed mode.")

    def generate(self, prompt, max_new_tokens=64):
        output = self.pipe(prompt, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"]

def main():
    print(f"CUDA GPUs available: {torch.cuda.device_count()}")

    # Only one actor is instantiated because device_map=auto distributes the model
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