import os
import ray
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@ray.remote
class InferenceWorker:
    """Ray Actor for distributed inference on HuggingFace CausalLM models."""

    def __init__(self, model_name: str, hf_token: str, gpus_per_worker: int = 1):
        """Initialize model and tokenizer for large models with model parallelism."""
        try:
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            self.is_instruction_model = "-it" in model_name or "instruct" in model_name.lower()
            
            n_gpus = torch.cuda.device_count()
            available_gpus = min(n_gpus, gpus_per_worker)

            # For big models, use device_map='auto' to slice across GPUs
            if available_gpus > 0:
                device_map = "auto"
                # Dynamically determine available GPU memory with a safety margin
                max_memory = {i: f"{int(torch.cuda.get_device_properties(i).total_memory / 1024**3 * 0.85)}GB" 
                              for i in range(available_gpus)}
            else:
                device_map = "cpu"
                max_memory = None

            # Use bfloat16 for Gemma-3 models if available for better performance/memory efficiency
            if "gemma-3" in model_name and torch.cuda.is_available() and hasattr(torch, 'bfloat16'):
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            print(f"Loading {model_name} with dtype={dtype}, device_map={device_map}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                token=hf_token,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Helps with large models
            )
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            ray.actor.exit_actor(f"Failed to initialize model: {str(e)}")

    def format_prompt(self, prompt: str) -> str:
        """Format the prompt based on model type."""
        if self.is_instruction_model:
            if "gemma-3" in self.model_name:
                # Gemma-3-IT specific formatting
                return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                # Generic instruction model formatting
                return f"USER: {prompt}\nASSISTANT: "
        return prompt

    def infer(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate a response from the model."""
        try:
            formatted_prompt = self.format_prompt(prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # Get the device of the first parameter of the model
            device = next(self.model.parameters()).device
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.1,
                )
            
            # Decode the full response and extract only the model's reply
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # For instruction models, extract only the model's reply
            if self.is_instruction_model:
                if "gemma-3" in self.model_name:
                    # Extract text after the model prompt
                    assistant_prefix = "<start_of_turn>model\n"
                    if assistant_prefix in full_output:
                        result = full_output.split(assistant_prefix, 1)[1]
                    else:
                        # Just return everything after the original prompt
                        result = full_output[len(formatted_prompt):]
                else:
                    # Generic instruction model - try to extract assistant response
                    if "ASSISTANT: " in full_output:
                        result = full_output.split("ASSISTANT: ", 1)[1]
                    else:
                        # Fallback to returning newly generated tokens
                        result = full_output[len(formatted_prompt):]
            else:
                # For non-instruction models, just return the newly generated tokens
                result = full_output[len(formatted_prompt):]
                
            return result.strip()
        except Exception as e:
            return f"Error during inference: {str(e)}"

def main():
    """Main entry - sets up Ray and processes distributed prompts."""
    try:
        # Initialize Ray with sensible defaults
        ray.init(ignore_reinit_error=True, runtime_env={"pip": ["transformers", "torch"]})

        # Get HuggingFace token from environment
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("The 'HF_TOKEN' environment variable is not set.")

        # Select the model to use
        model_name = "google/gemma-3-27b-it"
        
        # Sample prompts for testing
        prompts = [
            "Which language models support distributed inference?",
            "What is Ray and how does it help with distributed computing?",
            "Explain the concept of model parallelism in deep learning.",
            "Write a short story about a robot learning to paint."
        ]
        
        n_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {n_gpus}")
        
        if n_gpus == 0:
            print("Warning: No GPUs detected. Running on CPU only.")
        
        # For large models like gemma-3-27b, use more GPUs per worker if available
        gpus_per_worker = 4 if "27b" in model_name and n_gpus >= 4 else max(1, min(2, n_gpus))
        
        # Calculate optimal number of workers based on available GPUs
        num_workers = max(1, n_gpus // gpus_per_worker)
        
        print(f"Creating {num_workers} inference workers with {gpus_per_worker} GPUs per worker...")
        
        # Configure GPU resources for each worker
        InferenceWorkerWithGPUs = ray.remote(num_gpus=gpus_per_worker)(InferenceWorker)
        
        workers = [
            InferenceWorkerWithGPUs.remote(model_name, hf_token, gpus_per_worker)
            for _ in range(num_workers)
        ]

        # Dispatch prompts to workers cyclically
        print(f"Processing {len(prompts)} prompts...")
        prompt_tasks = [
            workers[i % num_workers].infer.remote(prompts[i], max_new_tokens=250, temperature=0.7)
            for i in range(len(prompts))
        ]
        
        # Process results as they become available
        for i, response in enumerate(ray.get(prompt_tasks)):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        # Shutdown Ray
        ray.shutdown()

if __name__ == "__main__":
    main()