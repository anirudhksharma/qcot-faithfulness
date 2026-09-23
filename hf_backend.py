"""GPU backend: real bitsandbytes quantization for Kaggle/Colab T4.

Loads ONE base checkpoint at FP16 / INT8 / INT4 so precision is the only
variable. This is the intended production backend (the GGUF/CPU one is a
laptop fallback). Interface matches the others: .generate(prompt) -> str
"""
import torch
import config


class HFBackend:
    def __init__(self, model_name: str, precision: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.precision = precision
        self.tok = AutoTokenizer.from_pretrained(model_name)

        kwargs = {"device_map": "auto"}
        if precision == "fp16":
            kwargs["dtype"] = torch.float16
        elif precision == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "int4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            raise ValueError(f"unknown precision {precision!r} (fp16|int8|int4)")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # Newer transformers return a BatchEncoding (dict) here, older ones a
        # bare tensor. return_dict=True normalizes to a dict we can **-unpack.
        enc = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,              # greedy, reproducible
            pad_token_id=self.tok.eos_token_id,
        )
        text = self.tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return text
