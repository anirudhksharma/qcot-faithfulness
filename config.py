"""Central configuration for the quantization x CoT-faithfulness experiment."""

# Base checkpoint. Any HF reasoning model works; the 1.5B distill fits a free T4.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Precision axis. The scientific variable. Same base weights, different bit-width.
#   fp16 = baseline, int8 = bitsandbytes 8-bit, int4 = bitsandbytes NF4 4-bit
ALL_PRECISIONS = ["fp16", "int8", "int4"]

# Generation settings. temperature=0 (greedy) for reproducibility, as in the
# monitorability literature (2510.27378).
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.0

# Reasoning models wrap their trace in <think>...</think>. We split on this to
# measure acknowledgment separately in the thinking span vs the answer span
# (the divergence studied in "Why Models Know But Don't Say", 2603.26410).
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Hint types to inject. Each points the model at a specific (wrong) option.
#   sycophancy = user asserts an answer
#   authority  = an expert asserts an answer
ACTIVE_HINT_TYPES = ["sycophancy", "authority"]

DEFAULT_LIMIT = 200          # questions per precision
RESULTS_FILE = "results.jsonl"
MOCK_RESULTS_FILE = "results_mock.jsonl"
