import os
import ray
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DistributedInference")

# Check if flash_attn is installed
has_flash_attn = importlib.util.find_spec("flash_attn") is not None
if not has_flash_attn:
    logger.warning("flash_attn is not installed. Falling back to standard attention implementation.")

# Keep the original InferenceWorker class for local model loading
class InferenceWorker:
    """Base class for distributed inference on HuggingFace CausalLM models."""

    def __init__(self, model_name: str, hf_token: str, gpus_per_worker: int = 1):
        """Initialize model and tokenizer for large models with model parallelism."""
        try:
            self.model_name = model_name
            
            # Special handling for Gemma-3 models
            if "gemma-3" in model_name.lower():
                logger.info(f"Using special configuration for Gemma-3 model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    token=hf_token,
                    trust_remote_code=True,  # Added for Gemma-3
                )
            else:
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

            logger.info(f"Loading {model_name} with dtype={dtype}, device_map={device_map}")
            
            # Special handling for Gemma-3 models
            if "gemma-3" in model_name.lower():
                # Set specific configuration options for Gemma-3
                model_kwargs = {
                    "torch_dtype": dtype,
                    "device_map": device_map,
                    "max_memory": max_memory,
                    "token": hf_token,
                    "trust_remote_code": True,
                    "revision": "main",  # Use main branch
                    "low_cpu_mem_usage": True,
                }
                
                # Only add flash_attention_2 if it's installed and we're on GPU
                if has_flash_attn and torch.cuda.is_available():
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using flash_attention_2 for faster inference")
                
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    max_memory=max_memory,
                    token=hf_token,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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
            logger.debug(f"Formatted prompt: {formatted_prompt}")
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # Get the device of the first parameter of the model
            device = next(self.model.parameters()).device
            logger.debug(f"Model is on device: {device}")
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Configure generation parameters
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
            }
            
            # Special handling for Gemma-3 models
            if "gemma-3" in self.model_name.lower():
                generation_config.update({
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                })
                
            logger.debug(f"Generation config: {generation_config}")
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
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
                
            # Ensure we return a primitive string type
            return str(result.strip())
        except Exception as e:
            logger.error(f"Error details in infer method: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error during inference: {str(e)}"


# Ray-compatible distributed worker implementation
class DistributedInferenceWorker:
    """Ray-compatible wrapper for distributed inference."""
    
    def __init__(self, model_name: str, hf_token: str, gpus_per_worker: int = 1):
        """Store initialization parameters only."""
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpus_per_worker = gpus_per_worker
        self.worker = None
        logger.info(f"DistributedInferenceWorker created for model {model_name}")
    
    def initialize(self):
        """Initialize the worker separately to avoid serialization issues."""
        if self.worker is None:
            try:
                logger.info("Initializing inference worker...")
                self.worker = InferenceWorker(
                    self.model_name, 
                    self.hf_token, 
                    self.gpus_per_worker
                )
                logger.info("Worker initialized successfully")
                return {"status": "success", "message": "Worker initialized successfully"}
            except Exception as e:
                error_msg = f"Failed to initialize worker: {str(e)}"
                logger.error(error_msg)
                import traceback
                traceback.print_exc()
                return {"status": "error", "message": error_msg}
        return {"status": "success", "message": "Worker already initialized"}
    
    def infer(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Pass the inference request to the local worker and ensure primitive return type."""
        # Initialize the worker if not already done
        if self.worker is None:
            init_result = self.initialize()
            if init_result["status"] == "error":
                return f"Error: {init_result['message']}"
        
        try:
            logger.info(f"Running inference with prompt: {prompt[:50]}...")
            result = self.worker.infer(prompt, max_new_tokens, temperature)
            logger.info("Inference completed successfully")
            # Ensure we're returning a primitive string type that Ray can easily serialize
            return str(result)
        except Exception as e:
            error_msg = f"Error in DistributedInferenceWorker.infer: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return f"Error during distributed inference: {str(e)}"


def main():
    """Main entry point - sets up Ray and processes distributed prompts."""
    try:
        # Initialize Ray with sensible defaults
        ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
        logger.info("Ray initialized")

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
        logger.info(f"Available GPUs: {n_gpus}")
        
        if n_gpus == 0:
            logger.warning("No GPUs detected. Running on CPU only.")
        
        # For large models like gemma-3-27b, determine optimal GPU allocation
        gpus_per_worker = min(n_gpus, 4 if "27b" in model_name else 1)
        
        # Create the right number of workers based on available GPUs
        num_workers = max(1, n_gpus // gpus_per_worker) if n_gpus > 0 else 1
        
        logger.info(f"Creating {num_workers} inference workers with {gpus_per_worker} GPUs per worker...")
        
        # Create Ray remote class with GPU requirements
        RemoteWorker = ray.remote(num_gpus=gpus_per_worker)(DistributedInferenceWorker)
        
        # Create workers
        workers = []
        for i in range(num_workers):
            worker = RemoteWorker.remote(model_name, hf_token, gpus_per_worker)
            workers.append(worker)
            logger.info(f"Created worker {i+1}/{num_workers}")
        
        # Initialize workers and check for errors
        init_results = ray.get([worker.initialize.remote() for worker in workers])
        for i, result in enumerate(init_results):
            if result["status"] == "error":
                logger.warning(f"Worker {i+1} initialization failed: {result['message']}")
        
        # Process prompts
        logger.info(f"Processing {len(prompts)} prompts...")
        prompt_tasks = [
            workers[i % num_workers].infer.remote(prompts[i], max_new_tokens=250, temperature=0.7)
            for i in range(len(prompts))
        ]
        
        # Process results as they become available
        for i, response in enumerate(ray.get(prompt_tasks)):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            print(f"Response: {response}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown Ray
        ray.shutdown()
        logger.info("Ray shutdown complete")


if __name__ == "__main__":
    main()