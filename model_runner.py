"""Model backends.

Two backends with the same interface (.generate(prompt) -> str):

- HFBackend: loads a real HF model at a given precision via bitsandbytes.
  Runs on a GPU (free Kaggle/Colab T4). This is what produces paper results.

- MockBackend: no GPU, no downloads. Produces deterministic <think>...</think>
  outputs whose behavior depends on precision, so the ENTIRE pipeline (loop,
  scorer, analysis) can be verified offline. It fabricates a plausible pattern
  (lower precision -> slightly less likely to verbalize the cue) purely so the
  plumbing can be exercised. Mock numbers are NOT results and are written to a
  separate file (results_mock.jsonl).
"""
import hashlib
from typing import Optional

import config


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #
class HFBackend:
    def __init__(self, model_name: str, precision: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.precision = precision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs = {"device_map": "auto"}
        if precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "int4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            raise ValueError(f"unknown precision {precision!r}")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=(config.TEMPERATURE > 0),
                temperature=config.TEMPERATURE if config.TEMPERATURE > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Mock backend (offline plumbing verification only)
# --------------------------------------------------------------------------- #
class MockBackend:
    def __init__(self, model_name: str, precision: str):
        self.precision = precision
        # A knob that makes lower precision slightly less likely to verbalize the
        # cue -- ONLY to prove the pipeline can detect a precision effect if one
        # exists. Not a scientific claim.
        self._ack_bias = {"fp16": 0.85, "int8": 0.70, "int4": 0.55}.get(precision, 0.7)

    def _rand(self, prompt: str) -> float:
        h = hashlib.sha256((self.precision + prompt).encode()).hexdigest()
        return (int(h[:8], 16) % 1000) / 1000.0

    def generate(self, prompt: str) -> str:
        r = self._rand(prompt)
        cued = "professor" in prompt.lower() or "pretty sure" in prompt.lower()

        # Pick a fabricated answer letter deterministically.
        letter = "ABCD"[int(self._rand(prompt[::-1]) * 4) % 4]

        if cued and r < 0.6:
            # Model "follows" the cue: extract the cued letter from the prompt.
            for tok in ("is A", "is B", "is C", "is D"):
                if tok in prompt:
                    letter = tok[-1]
                    break
            if r < self._ack_bias:
                think = ("The user gave a hint. The professor / user thinks it is "
                         f"{letter}. That influences me, so I'll go with it.")
            else:
                think = f"Let me reason. I conclude it is {letter}."
        else:
            think = f"Working it out independently, I believe it is {letter}."

        return f"{config.THINK_OPEN}{think}{config.THINK_CLOSE} The answer is ({letter})."


def make_backend(model_name: str, precision: str, mock: bool):
    return (MockBackend if mock else HFBackend)(model_name, precision)
