import ray
import os
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

def inference_loop_per_worker(config):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be provided in config or environment!")

    os.environ["HF_TOKEN"] = hf_token

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
        use_auth_token=hf_token
    )

    prompt = config.get("prompt", "Hello, world!")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return for Ray aggregation
    return {"output": result}

if __name__ == "__main__":
    ray.init()

    HF_TOKEN = os.environ.get("HF_TOKEN")
    config = {
        "model_name": "google/gemma-7b",
        "prompt": "Which language models support distributed inference?",
        "hf_token": HF_TOKEN
    }

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

    trainer = TorchTrainer(
        train_loop_per_worker=inference_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config
    )

    # Run and collect results (from all workers)
    result = trainer.fit()
    # Access per-worker return values
    if hasattr(result, "metrics") and "output" in result.metrics:
        # Single worker
        outputs = [result.metrics["output"]]
    elif hasattr(result, "metrics") and "output" not in result.metrics and "per_worker_metrics" in result.metrics:
        # Multi-worker (older versions)
        outputs = [m["output"] for m in result.metrics["per_worker_metrics"]]
    else:
        # Air 2.x: results are in result.metrics or result.metrics["output"] or result.metrics["outputs"]
        outputs = result.metrics.get("outputs", None) or result.metrics.get("output", None)
        if outputs is not None and not isinstance(outputs, list):
            outputs = [outputs]

    print("All worker outputs:", outputs)