import os
import ray
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@ray.remote(num_gpus=torch.cuda.device_count())
class InferenceWorker:
    """Ray Actor for distributed inference on HuggingFace CausalLM models."""

    def __init__(self, model_name: str, hf_token: str):
        """Initialize model and tokenizer for large models with model parallelism."""

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        n_gpus = torch.cuda.device_count()

        # For big models, use device_map='auto' to slice across all GPUs
        # Set per-GPU max memory if needed, e.g. {"0": "40GB", "1": "40GB", ...}
        if n_gpus > 1:
            device_map = "auto"
            max_memory = {i: "40GB" for i in range(n_gpus)}
        elif n_gpus == 1:
            device_map = 0  # Single GPU
            max_memory = None
        else:
            device_map = "cpu"
            max_memory = None

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            token=hf_token,
            trust_remote_code=True,  # Optional: required for some big models.
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
    """Main entry - sets up Ray and processes distributed prompts."""
    ray.init()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("The 'HF_TOKEN' environment variable is not set.")

    model_name = "google/gemma-7b"
    prompts = [
        "Which language models support distributed inference?",
        "What is Ray?",
    ]
    n_gpus = torch.cuda.device_count()
    num_workers = 1  # For big models, usually 1 worker per big model (per node)

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