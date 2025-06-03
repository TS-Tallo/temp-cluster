import ray
import os
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure your HF_TOKEN is set — recommend export HF_TOKEN=xxx in shell
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def inference_loop_per_worker(config):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Get the Hugging Face token from environment for every worker
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable not set on worker!")

    # Change as needed for your model
    model_name = config["model_name"]

    print("Initializing model on worker...")
    # tokenizer and model will be loaded using the HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
        use_auth_token=hf_token
    )

    print(f"Worker device count: {torch.cuda.device_count()}")
    prompt = config.get("prompt", "Hello, world!")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Sample output:", result)
    return {"output": result}

if __name__ == "__main__":
    # Ensure Ray is initialized. This will use Ray cluster if available.
    ray.init()

    # Pass config for your run
    config = {
        "model_name": "google/gemma-7b",
        "prompt": "Which language models support distributed inference?"
    }

    # ScalingConfig to define cluster usage (e.g., 2 workers, 1 GPU each)
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

    # Pass the env var with the token to workers (if not globally exported)
    # You can also do os.environ['HF_TOKEN'] = ... here
    trainer = TorchTrainer(
        train_loop_per_worker=inference_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=ray.air.RunConfig(env={"HF_TOKEN": HF_TOKEN})
    )

    # Run distributed inference/train — returns results from all workers
    results = trainer.fit()
    print("All worker results:", results)