import ray
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@ray.remote(num_gpus=1)
class InferenceWorker:
    def __init__(self, model_name, hf_token):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        device_map = "auto" if torch.cuda.device_count() > 1 else None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            use_auth_token=hf_token
        )
        # Always get device for later use
        self.device = next(self.model.parameters()).device

    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)
        # To avoid .cpu() errors: decode handles CUDA tensors automatically, but safely move if needed
        result = self.tokenizer.decode(outputs[0].cpu() if outputs[0].is_cuda else outputs[0], skip_special_tokens=True)
        return result

if __name__ == "__main__":
    ray.init()

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("Please set the HF_TOKEN environment variable for HuggingFace model access.")

    MODEL_NAME = "google/gemma-7b"

    prompts = [
        "Which language models support distributed inference?",
        "What is Ray?",
    ]
    # Create as many actors as you want distributed in the cluster
    num_workers = 2
    workers = [InferenceWorker.remote(MODEL_NAME, HF_TOKEN) for _ in range(num_workers)]

    # Assign one prompt to each worker (cyclic/repeat if needed)
    prompt_tasks = [
        workers[i % num_workers].infer.remote(prompts[i])
        for i in range(len(prompts))
    ]
    responses = ray.get(prompt_tasks)
    for i, response in enumerate(responses):
        print(f"Response from worker {i}: {response}")