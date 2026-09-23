"""GGUF backend for CPU inference via llama-cpp-python.

Lets the SAME experiment run on CPU with no GPU: the quantization axis becomes
GGUF quant levels (F16 -> Q8_0 -> Q4_K_M) of one base checkpoint. Same
scientific comparison as bitsandbytes INT8/INT4, just CPU-friendly.

Interface matches model_runner: .generate(prompt) -> str
"""
import os
import config

# Map a precision tag to its downloaded GGUF file.
GGUF_FILES = {
    "f16": "models/Qwen2.5-1.5B-Instruct-f16.gguf",
    "q8":  "models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
    "q4":  "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
}


class GGUFBackend:
    def __init__(self, model_name: str, precision: str):
        from llama_cpp import Llama
        path = GGUF_FILES.get(precision)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"no GGUF for precision {precision!r} at {path}")
        self.precision = precision
        self.llm = Llama(
            model_path=path,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4,
            verbose=False,
            seed=42,
        )

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=0.0,   # greedy for reproducibility
            seed=42,
        )
        return out["choices"][0]["message"]["content"]
