import os
import ray
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@ray.remote(num_gpus=1)
class InferenceWorker:
    """Ray Actor for distributed inference on HuggingFace CausalLM models."""

    def __init__(self, model_name: str, hf_token: str):
        """Initialize model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        device_map = "auto" if torch.cuda.device_count() > 1 else None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            token=hf_token
        )

    def infer(self, prompt: str) -> str:
        """Generate a response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

def main():
    """Main entry - sets up Ray and crawls distributed prompts."""
    ray.init()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("The 'HF_TOKEN' environment variable is not set.")

    model_name = "google/gemma-7b"
    prompts = [
        "Which language models support distributed inference?",
        "What is Ray?",
    ]
    num_workers = 2
    workers = [
        InferenceWorker.remote(model_name, hf_token)
        for _ in range(num_workers)
    ]

    # Dispatch prompts to workers cyclically
    prompt_tasks = [
        workers[i % num_workers].infer.remote(prompts[i])
        for i in range(len(prompts))
    ]
    responses = ray.get(prompt_tasks)
    for i, response in enumerate(responses):
        print(f"Response from worker {i}: {response}")

if __name__ == "__main__":
    main()