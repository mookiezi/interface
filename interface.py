"""
interface.py - Discord-Hermes Model Chat Interface
Description:
    This script provides an interactive CLI for chatting with Hugging Face language models,
    designed for Discord-style conversational flow. It supports both Transformers and GGUF
    (llama.cpp) models, with optional LoRA adapter loading and bitsandbytes quantization.
Key Features & Quick Controls:
    Keyboard Shortcuts:
        • Enter      → submit current input
        • Ctrl+T     → insert newline into the current prompt (multi-line messages)
        • Ctrl+C     → cancel mid-generation or mid-stream and keep partial output
        • Ctrl+Z     → exit cleanly, freeing VRAM and memory
    Commands:
        • /clear (/c, /reset, !clear, !c)   → reset all context
        • /back (/b, !b)                    → undo last user+assistant exchange and preview history
        • /h [N] (!h)                       → enable history using last N exchanges (default: all)
        • /d                                → disable history
        • On-the-fly parameter tuning:
            /min N     → set min new tokens
            /max N     → set max new tokens
            /temp X    → set temperature
            /p X       → set top-p
            /k N       → set top-k
        • Randomization:
            /r         → randomize params
            /rh        → randomize with high variance
        • /stop                              → toggle stopping further extension mid-generation
    Generation Flow:
        • Extension flow: prefer short replies (min–max tokens) but extend until EOS for natural endings
        • Configurable prompt modes (system prompt, assistant prompt, blank mode)
    Advanced:
        • LoRA stacking: frozen base adapter + active adapter support
        • Supports GGUF (llama.cpp) with selectable chat templates (--gguf-chat-format)
        • Optional code detection and filtering with auto-reprompt
Arguments:
    -m, --model                     Model path or Hugging Face repo ID (default: mookiezi/Discord-Micae-Hermes-3-3B)
    -q, --quant                     Quantization mode: 4 or 8 (default: off). Use `-q` (no value) for 4-bit, or `-q 8` for 8-bit
    -fl, --frozen-lora              Model path or Hugging Face repo ID of the base LoRa adapter to load and freeze
    -c, --checkpoint                Model path or Hugging Face repo ID of the LoRa adapter to load
    -chs, --checkpoint-subfolder    Subfolder of the path or Hugging Face repo ID of the LoRa adapter to load
    --deephermes                    Enable DeepHermes formatting instead of ChatML
    --gguf                          Use GGUF model format with llama.cpp backend
    --gguf-chat-format              Chat format for GGUF models (default: "chatml")
    --blank                         Raw user input only, no prompts/system context
    -asc, --assistant-system-combo  Include both system and assistant system prompts
    -as, --assistant-system         Use assistant system prompt instead of standard
    --just-system-prompt            Use only the system prompt with user input
    --no-system-prompt              Do not include system prompt
    --no-assistant-prompt           Do not include assistant prompt
    --code-check                    Enable code detection and filtering via classifier
    -au, --auto                     Run preset inputs (hello → what do you do → wow tell me more) 5 times with /clear in between, then exit
Usage (quick help):
    python interface.py -h
USAGE / RECIPES:
  Basic (Transformers, full precision):
    python interface.py -m mookiezi/Discord-Micae-Hermes-3-3B
  Quantization (Transformers):
    # 4-bit:
    python interface.py -m repo -q
    # 8-bit:
    python interface.py -m repo -q 8
    # full precision:
    python interface.py -m repo
  GGUF (llama.cpp backend):
    python interface.py --gguf -m /path/to/model.gguf --gguf-chat-format chatml
    # alternate chat template:
    python interface.py --gguf -m /path/to/model.gguf --gguf-chat-format alpaca
  LoRA (frozen base + active adapter):
    python interface.py -m base/model \
      -fl path/to/frozen_base_lora \
      -c path/to/active_adapter --checkpoint-subfolder adapter_subdir
  Prompt modes:
    # Raw user input, no system/assistant prompts:
    python interface.py --blank
    # Assistant system prompt instead of standard:
    python interface.py --assistant-system
    # System + assistant system combined:
    python interface.py --assistant-system-combo
    # Just the system prompt (no assistant prompt preface):
    python interface.py --just-system-prompt
    # Strip system prompt entirely:
    python interface.py --no-system-prompt
    # Strip assistant prompt entirely:
    python interface.py --no-assistant-prompt
  Format toggle:
    # Use DeepHermes formatting instead of ChatML:
    python interface.py --deephermes
  Auto run demo + exit:
    python interface.py --auto
"""

# MIT License
# 
# Copyright (c) 2025 mookziei
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, AutoConfig
from peft import PeftModel
from huggingface_hub import login
import argparse
import logging
import re
from transformers import TextStreamer
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from huggingface_hub import login
import random
from transformers import LogitsProcessor
from transformers import LogitsProcessorList
import signal
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

parser = argparse.ArgumentParser(description="HuggingFace Model Chat Interface")

parser.add_argument("-au", "--auto", action="store_true",
                    help="Run preset inputs (hello → what do you do → wow tell me more) 5 times with /clear in between, then exit")
parser.add_argument("-m", "--model", default="mookiezi/Discord-Micae-Hermes-3-3B",
                    help="Model path or Hugging Face repo ID")
parser.add_argument(
    "-q", "--quant",
    nargs="?",
    choices=("4", "8"),
    const="4",
    help="Quantization mode: 4 or 8. Omit this flag for full precision. Using `-q` alone selects 4-bit."
)
parser.add_argument("-fl", "--frozen-lora",
                    help="Model path or Hugging Face repo ID of the base LoRa adapter to load and freeze")
parser.add_argument("-c", "--checkpoint",
                    help="Model path or Hugging Face repo ID of the LoRa adapter to load")
parser.add_argument("-chs", "--checkpoint-subfolder",
                    help="Subfolder of the path or Hugging Face repo ID of the LoRa adapter to load")
parser.add_argument("--deephermes", action="store_true",
                    help="Enable DeepHermes formatting instead of ChatML")
parser.add_argument("--gguf", action="store_true",
                    help="Use GGUF model format (llama.cpp backend)")
parser.add_argument("--gguf-chat-format", default="chatml",
                    help='Chat format for GGUF models (default: "chatml")')
parser.add_argument("--blank", action="store_true",
                    help="Use only raw user input (no prompts/system context)")
parser.add_argument("-asc", "--assistant-system-combo", action="store_true",
                    help="Include both system and assistant system prompts")
parser.add_argument("-as", "--assistant-system", action="store_true",
                    help="Use assistant system prompt instead of standard system prompt")
parser.add_argument("--just-system-prompt", action="store_true",
                    help="Use only the system prompt with user input")
parser.add_argument("--no-system-prompt", action="store_true",
                    help="Do not include system prompt")
parser.add_argument("--no-assistant-prompt", action="store_true",
                    help="Do not include assistant prompt")
parser.add_argument("--code-check", action="store_true",
                    help="Enable code detection and filtering via classifier")

args = parser.parse_args()

# Apply args to variables
DEEPHERMES = args.deephermes
GGUF = args.gguf
GGUF_CHAT_FORMAT = args.gguf_chat_format
BLANK = args.blank
ASSISTANT_SYSTEM_COMBO = args.assistant_system_combo
ASSISTANT_SYSTEM = args.assistant_system
JUST_SYSTEM_PROMPT = args.just_system_prompt
NO_SYSTEM_PROMPT = args.no_system_prompt
NO_ASSISTANT_PROMPT = args.no_assistant_prompt
CODE_CHECK = args.code_check
QUANTIZATION = args.quant or "off"
USE_QUANT  = QUANTIZATION in ("4", "8")
# ================================
base_model_name = args.model
FROZEN_LORA_PATH = args.frozen_lora
checkpoint_path = args.checkpoint or ""
checkpoint_subfolder = args.checkpoint_subfolder
USE_BASE_MODEL_ONLY = not (FROZEN_LORA_PATH or checkpoint_path)
# ================================
PLAIN_SYSTEM_PROMPT = """"""
ASSISTANT_SYSTEM_PROMPT = """"""
# ================================
if args.auto:
    MIN_NEW_TOKENS = 4
    MAX_NEW_TOKENS = 60
    TEMPERATURE = 0.6
    TOP_P = 0.9
    TOP_K = 55
else:
    MIN_NEW_TOKENS = random.randint(3, 5)
    MAX_NEW_TOKENS = random.randint(40, 75)
    TEMPERATURE = random.uniform(0.5, 0.9)
    TOP_P = random.uniform(0.7, 0.9)
    TOP_K = random.randint(40, 50)
# ================================

STOP_EXTENSION = False

# --- Prefer 40–75 tokens, but extend until <|im_end|> (or eos) ---
def generate_prefer_short_then_extend(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    first_min=MIN_NEW_TOKENS,
    first_max=MAX_NEW_TOKENS,
    extend_step=64,
    hard_cap=1024,
    **gen_kwargs,
):
    import random, torch

    # Resolve EOS ids (<|im_end|> for ChatML/Hermes, or tokenizer.eos_token_id)
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None:
            eos_ids.append(im_end_id)
    except Exception:
        pass
    eos_ids = [eid for eid in eos_ids if eid is not None]
    if not eos_ids:
        raise ValueError("No EOS token id found. Make sure '<|im_end|>' exists or set tokenizer.eos_token.")

    # Strip keys we set manually so no duplication happens
    for key in ["max_new_tokens", "eos_token_id", "do_sample", "pad_token_id"]:
        gen_kwargs.pop(key, None)

    start_len = input_ids.shape[1]
    seq = input_ids
    attn = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

    budget = random.randint(first_min, first_max)

    while True:
        out = model.generate(
            input_ids=seq,
            attention_mask=attn,
            max_new_tokens=budget,
            eos_token_id=eos_ids[0],
            pad_token_id=tokenizer.pad_token_id or eos_ids[0],
            return_dict_in_generate=True,
            **gen_kwargs,
        )
        seq = out.sequences
        attn = torch.ones_like(seq)

        new_tokens = seq[0, start_len:].tolist()
        if any(t in eos_ids for t in new_tokens):
            break

        if len(new_tokens) >= hard_cap:
            break

        if globals().get("STOP_EXTENSION", False):
            break

        budget = extend_step

    return seq



if checkpoint_path is None:
    USE_BASE_MODEL_ONLY = True
    checkpoint_path = ""
else:
    USE_BASE_MODEL_ONLY = False

if ASSISTANT_SYSTEM:
    if DEEPHERMES:
        SYSTEM_PROMPT = f"<|start_header_id|>assistant<|end_header_id|>\n{PLAIN_SYSTEM_PROMPT}"
    else:
        SYSTEM_PROMPT = f"<|im_start|>assistant\n{PLAIN_SYSTEM_PROMPT}<|im_end|>"
elif ASSISTANT_SYSTEM_COMBO:
    if DEEPHERMES:
        SYSTEM_PROMPT = f"<|start_header_id|>system<|end_header_id|>\n{PLAIN_SYSTEM_PROMPT}<|start_header_id|>assistant<|end_header_id|>\n{ASSISTANT_SYSTEM_PROMPT}"
    else:
        SYSTEM_PROMPT = f"<|im_start|>system\n{PLAIN_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>assistant\n{ASSISTANT_SYSTEM_PROMPT}<|im_end|>"
else:
    if DEEPHERMES:
        SYSTEM_PROMPT = f"<|start_header_id|>system<|end_header_id|>\n{PLAIN_SYSTEM_PROMPT}"
    else:
        SYSTEM_PROMPT = f"<|im_start|>system\n{PLAIN_SYSTEM_PROMPT}<|im_end|>"


def handle_sigstp(signum, frame):
    print("\n\033[1;91m[Received Ctrl+Z — exiting.]\033[0m")
    torch.cuda.empty_cache()
    exit(0)

signal.signal(signal.SIGTSTP, handle_sigstp)

class RecentTokenBlocker(LogitsProcessor):
    def __init__(self, window_size: int, eos_id: int | None = None):
        self.window_size = window_size
        self.eos_id = eos_id

    def __call__(self, input_ids, scores):
        # Penalize last N tokens (use a big negative instead of -inf)
        recent_tokens = input_ids[0, -self.window_size:].tolist()
        if recent_tokens:
            for token_id in set(recent_tokens):
                if 0 <= token_id < scores.shape[-1]:
                    scores[0, token_id] = -1e9

        # Sanitize logits
        scores = torch.nan_to_num(scores, neginf=-1e9)

        # If everything is blocked, fallback to EOS
        if torch.all(scores[0] <= -9e8):
            scores[0].fill_(-1e9)
            if self.eos_id is not None and 0 <= self.eos_id < scores.shape[-1]:
                scores[0, self.eos_id] = 0

        return scores
    
processor_list = LogitsProcessorList()
processor_list.append(RecentTokenBlocker(window_size=3))

def randomize():
    global MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K
    MIN_NEW_TOKENS = random.randint(3, 5)
    MAX_NEW_TOKENS = random.randint(40, 75)
    TEMPERATURE = random.uniform(0.5, 0.9)
    TOP_P = random.uniform(0.7, 0.9)
    TOP_K = random.randint(40, 75)
    
    return

def randomize_high_variance():
    global MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K
    MIN_NEW_TOKENS = random.randint(3, 20)
    MAX_NEW_TOKENS = random.randint(40, 150)
    TEMPERATURE = random.uniform(0.2, 0.9)
    TOP_P = random.uniform(0.7, 1)
    TOP_K = random.randint(40, 150)
    
    return

logging.getLogger("transformers").setLevel(logging.ERROR)

bnb_config4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
bnb_config8bit = BitsAndBytesConfig(load_in_8bit=True)

bnb_config = None
if USE_QUANT:
    bnb_config = bnb_config4bit if QUANTIZATION == "4" else bnb_config8bit

bindings = KeyBindings()
session = PromptSession(key_bindings=bindings)

@bindings.add('enter')
def _(event):
    event.app.exit(result=event.app.current_buffer.text)

@bindings.add('c-t')
def _(event):
    event.app.current_buffer.insert_text('\n')

tokenizer = None  # define early for fallback

if GGUF:
    from llama_cpp import Llama
    model = Llama(
        model_path=base_model_name,
        n_gpu_layers=999,          # same as -ngl 999
        n_ctx=4096,                # same as -c 4096
        n_batch=2048,              # same as -b 2048
        n_threads=12,              # same as -t 12
        chat_format=GGUF_CHAT_FORMAT,
        use_mmap=True,             # fastest load
        logits_all=False
    )
    tokenizer = None
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # ---- normalize config BEFORE ANY model load ----
    cfg = AutoConfig.from_pretrained(base_model_name)
    if getattr(cfg, "hidden_act", None) == "swiglu":
        cfg.hidden_act = "silu"

    if USE_BASE_MODEL_ONLY:
        if USE_QUANT and bnb_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                config=cfg,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                config=cfg,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        model.resize_token_embeddings(len(tokenizer))

    else:
        # Load base first (for LoRA stack)
        if USE_QUANT and bnb_config is not None:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                config=cfg,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                config=cfg,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        base_model.resize_token_embeddings(len(tokenizer))

        # Load frozen adapter first (if present)
        if FROZEN_LORA_PATH:
            base_model = PeftModel.from_pretrained(
                base_model,
                FROZEN_LORA_PATH,
                is_trainable=False
            )
            print(f"Loaded frozen LoRA adapter from {FROZEN_LORA_PATH}")

        # Active adapter (if provided)
        if checkpoint_path:
            model = PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                subfolder=checkpoint_subfolder if checkpoint_subfolder else None,
            )
        else:
            model = base_model

class CyanPromptStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first = True
        self.output = ""
        self._buffer = []

    # Receive raw, possibly-unstable pieces → just buffer
    def on_text(self, text, **kwargs):
        self._buffer.append(text)

    # Receive stable text spans → clean once, then print
    def on_finalized_text(self, text, **kwargs):
        text = "".join(self._buffer) + text
        self._buffer.clear()

        text = strip_begin_of_text(text)
        text = clean_special_tokens(text)
        #text = filter_emojis_keep_first_single(text)
        #text = remove_all_emojis(text)

        self.output += text
        if self.first:
            #print("\033[1;96m> \033[0m", end="", flush=True)
            self.first = False
        print(text, end="", flush=True)

    def on_final_text(self, text, **kwargs):
        # final flush if anything remains
        if self._buffer:
            self.on_finalized_text("", **kwargs)
        self.output += text
        print(text, end="", flush=True)

    def end(self):
        super().end()
        self.first = True

def filter_emojis_keep_first_single(text):
    # Emoji regex pattern (single emojis)
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U0001F018-\U0001F270"
        "\U0001F650-\U0001F67F"
        "\U00002600-\U000026FF"
        "\U00002300-\U000023FF"
        "]+",
        flags=re.UNICODE
    )

    # Pattern for emoji groups: 2 or more consecutive emojis
    emoji_group_pattern = re.compile(r'(' + emoji_pattern.pattern + r'){2,}', flags=re.UNICODE)

    # Remove all emoji groups first (clusters of 2+ emojis)
    text = emoji_group_pattern.sub("", text)

    # Now remove all single emojis except the first one
    first_emoji_found = False
    def replace_single_emoji(match):
        nonlocal first_emoji_found
        if not first_emoji_found:
            first_emoji_found = True
            return match.group(0)  # keep first emoji
        return ""  # remove all others

    filtered_text = emoji_pattern.sub(replace_single_emoji, text)

    return filtered_text

def remove_all_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U0001F018-\U0001F270"
        "\U0001F650-\U0001F67F"
        "\U00002600-\U000026FF"
        "\U00002300-\U000023FF"
        "]+",
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'.', text)

def show_params():
    print(
        f"\033[1;92m[Params]"
        f" min={MIN_NEW_TOKENS} | max={MAX_NEW_TOKENS} |"
        f" temp={TEMPERATURE:.2f} | p={TOP_P:.2f} | k={TOP_K}\033[0m"
    )

if CODE_CHECK:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

if checkpoint_path:
    folder_name = f"{base_model_name.replace('/', '_')}_{os.path.basename(checkpoint_path).replace('/', '_')}"
else:
    folder_name = base_model_name.replace('/', '_')
output_dir = os.path.join(os.getcwd(), "interface_output", folder_name)
os.makedirs(output_dir, exist_ok=True)
base_filename = f"{folder_name}.txt"
output_path = os.path.join(output_dir, base_filename)

if args.auto:
    base_filename = "auto.txt"
    output_path = os.path.join(output_dir, base_filename)
else:
    base_filename = f"{folder_name}.txt"
    output_path = os.path.join(output_dir, base_filename)

    if os.path.exists(output_path):
        i = 1
        while True:
            numbered = os.path.join(output_dir, f"{folder_name}_{i}.txt")
            if not os.path.exists(numbered):
                output_path = numbered
                break
            i += 1

if tokenizer:
    if DEEPHERMES:
        reserved_special_ids = [
            tid for tok, tid in tokenizer.get_vocab().items()
            if re.match(r"<\|reserved_special_token_\d+\|>", tok)
        ]
        im_end_token_id = [tokenizer.convert_tokens_to_ids("<|eot_id|>")] + reserved_special_ids
    else:
        im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
else:
    im_end_token_id = None

def get_generation_params():
    return {
        "do_sample": True,
        "eos_token_id": (
            ([im_end_token_id] if isinstance(im_end_token_id, int) else (im_end_token_id or []))
            + ([tokenizer.eos_token_id] if tokenizer and tokenizer.eos_token_id is not None else [])
        ) or None,
        "bos_token_id": tokenizer.bos_token_id if tokenizer else None,
        "min_new_tokens": MIN_NEW_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        #"pad_token_id": tokenizer.eos_token_id if tokenizer else None,
        "logits_processor": processor_list,
    }

chat_history = []

def cleanup_and_exit():
    print("\n\033[1;91m[Cleaning up — freeing VRAM]\033[0m")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    raise SystemExit(0)

def log(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(colorize(text))
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}]\n{text}\n\n────────────────────────────────────────────────────────────────────────────\n\n")

def justlog(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}]\n{text}\n\n────────────────────────────────────────────────────────────────────────────\n\n")

def clean_special_tokens(text: str) -> str:
    text = re.sub(r"<\|im_start\|>", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|eot_id\|>", "", text)
    text = re.sub(r"<\|reserved_special_token_\d+\|>", "", text)
    text = re.sub(r"\n+\s*(<\|start_header_id\|>assistant<\|end_header_id\|>)", r"\n\1", text)
    return text

def count_tokens(s):
    if tokenizer:
        return len(tokenizer.encode(s))
    else:
        return len(s.split())  # fallback for GGUF

def colorize(text):
    text = text.replace("<|im_start|>system", "\033[1;93m<|im_start|>system\033[0m")
    text = text.replace("<|im_start|>user", "\033[1;96m<|im_start|>user\033[0m")
    text = text.replace("<|im_start|>assistant", "\033[1;95m<|im_start|>assistant\033[0m")
    text = text.replace("<|im_end|>", "\033[1;1;90m<|im_end|>\033[0m")
    return text

def extract_assistant_reply(full_text):
    if DEEPHERMES:
        start_token = "<|start_header_id|>assistant<|end_header_id|>"
        start_idx = full_text.rfind(start_token)
        if start_idx == -1:
            return ""
        start_idx += len(start_token)

        # Find the earliest end among <|eot_id|> or any <|reserved_special_token_N|>
        end_pattern = re.compile(r"<\|eot_id\|>|<\|reserved_special_token_\d+\|>")
        m = end_pattern.search(full_text, start_idx)
        end_idx = m.start() if m else len(full_text)  # ← fallback to EOF
    else:
        start_token = "<|im_start|>assistant"
        end_token = "<|im_end|>"
        start_idx = full_text.rfind(start_token)
        if start_idx == -1:
            return ""
        start_idx += len(start_token)
        end_idx = full_text.find(end_token, start_idx)
        if end_idx == -1:
            end_idx = len(full_text)  # ← fallback to EOF

    return full_text[start_idx:end_idx].strip()

def lowercase_lines_and_sentences(text: str) -> str:
    def should_skip(word: str) -> bool:
        return len(word) >= 2 and word.isupper()

    # Lowercase start of every line (unless all caps and >= 2 chars)
    def lower_line_start(line: str) -> str:
        if not line.strip():   # covers empty or whitespace-only
            return ''
        parts = line.split(maxsplit=1)
        first_word = parts[0]
        if should_skip(first_word):
            return line
        # Rebuild line if split > 1, else just lower first char
        if len(parts) > 1:
            rest = parts[1]
            return first_word[:1].lower() + first_word[1:] + ' ' + rest
        return line[:1].lower() + line[1:]
    
    # Lowercase start of every sentence after . ? !
    def lower_after_punct(match):
        punct_space = match.group(1)
        rest = match.group(2)
        first_word = rest.split()[0]
        if should_skip(first_word):
            return punct_space + rest
        return punct_space + rest[:1].lower() + rest[1:]
    
    return re.sub(r'([.?!]\s+)(\w+)', lower_after_punct, text)

def strip_begin_of_text(s):
    prefix = "<|begin_of_text|>"
    if s.startswith(prefix):
        return s[len(prefix):]
    return s

def strip_code_word(text: str) -> str:
    import re
    words_to_remove = [
        'api', 'program', 'programmer', 'script', 'scripts', 'code', 'coded', 'codeblock',  'coding', 'log', 'logs', 'snippet', 'function',
        'script', 'algorithm', 'web', 'library', 'python', 'bot' , 'nodejs'
    ]
    pattern = r'\b(?:' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)

def classify(text, threshold=0.8):
    if not text.strip():
        return False, {}
    candidate_labels = ["code", "chat"]
    result = classifier(text, candidate_labels)
    scores = dict(zip(result['labels'], result['scores']))
    is_code = scores.get("code", 0) > scores.get("chat", 0) and scores["code"] > threshold
    return is_code, scores


# Initialize dummy values for current_prompt and user_input for the first generation_params
current_prompt = ""
user_input = ""
generation_params = get_generation_params()

if args.auto:
    with open(output_path, "a", encoding="utf-8") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}]")
        f.write(f"# model: {base_model_name}\n")
        f.write(f"# frozen-lora: {FROZEN_LORA_PATH or 'None'}\n")
        f.write(f"# checkpoint: {checkpoint_path or 'None'}\n")
        f.write(f"# args: {vars(args)}\n")
        f.write(f"# generation_params: {generation_params}\n\n────────────────────────────────────────────────────────────────────────────\n\n")

else:
    with open(output_path, "w", encoding="utf-8") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}]")
        f.write(f"# model: {base_model_name}\n")
        f.write(f"# frozen-lora: {FROZEN_LORA_PATH or 'None'}\n")
        f.write(f"# checkpoint: {checkpoint_path or 'None'}\n")
        f.write(f"# args: {vars(args)}\n")
        f.write(f"# generation_params: {generation_params}\n\n────────────────────────────────────────────────────────────────────────────\n\n")

max_retries = 5
history_enabled = True
cot_max_exchanges = 1000
cached_history_size = 1000

if not GGUF:
    model.eval()
    aa = getattr(model, "active_adapter", None)
    if aa is not None:
        print(aa)

else:
    print(f"🚀 Model launched ({base_model_name})")


turn = 0

if args.auto:
    cycle = ["hello", "what do you do", "wow tell me more", "/clear"]
    auto_inputs = cycle * 5  # repeat cycle 3 times
else:
    auto_inputs = []

#=============
# START LOOP
#=============

while True:
    generation_params = get_generation_params()
    show_params()

    # print("\033[1;96m>  ", end="", flush=True)

    if auto_inputs:
        user_input = auto_inputs.pop(0)
        print(f"\033[1;96m> {user_input}\033[0m")  # show it like a typed input
    else:
        user_input = session.prompt(multiline=True)
    user_input = user_input.strip()

    cmd = user_input.lower()

   # --- clear/reset first (includes /c and !c aliases) ---
    if cmd in ["/clear", "!clear", "/reset", "!reset", "/c", "!c"]:
        chat_history.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print("\033[1;93m[Chat history cleared.]\033[0m")
        justlog("[Chat history cleared.]")
        # If we're in auto mode and this was the final queued item, exit now.
        if args.auto and not auto_inputs:
            print("\n\033[1;92m[Auto mode finished — exiting.]\033[0m")
            cleanup_and_exit()
        continue

    # --- /back (/b): undo last exchange and reprint tail ---
    if cmd in ["/back", "!back", "/b", "!b"]:
        if chat_history:
            u, a = chat_history.pop()
            turn = max(0, turn - 1)
            print("\033[1;93m[Undo] Removed last user+assistant exchange.\033[0m")
            justlog("[Undo] Removed last user+assistant exchange.]")

            TAIL = 3  # how many exchanges to preview
            tail = chat_history[-TAIL:]
            if tail:
                if DEEPHERMES:
                    preview = "\n".join(
                        f"<|start_header_id|>user<|end_header_id|>\n{uu}\n"
                        f"<|start_header_id|>assistant<|end_header_id|>\n{aa}<|eot_id|>"
                        for uu, aa in tail
                    )
                else:
                    preview = "\n".join(
                        f"<|im_start|>user\n{uu}<|im_end|>\n"
                        f"<|im_start|>assistant\n{aa}<|im_end|>"
                        for uu, aa in tail
                    )
                print(colorize(preview))
            else:
                print("\033[1;90m[Chat history is now empty.]\033[0m")
        else:
            print("\033[1;90m[Nothing to undo.]\033[0m")
        continue

    # --- /h (CoT on, optional count) ---
    m = re.match(r"^(/h|!h)(\s+(\d+))?$", cmd)
    if m:
        history_enabled = True
        cot_max_exchanges = int(m.group(3)) if m.group(3) else len(chat_history)
        print(f"\033[1;94m[History] Using last {cot_max_exchanges} exchanges.]\033[0m")
        justlog(f"[Chat history] Using last {cot_max_exchanges} exchanges.]")
        continue

    # --- /d (CoT off) ---
    if cmd in ["/d", "!d"]:
        history_enabled = False
        cot_max_exchanges = 0
        print("\033[1;94m[Chat history disabled.]\033[0m")
        justlog("[Chat history disabled.]")
        continue

    # === Multi-param updates: allow "/k 40 /t 1 /p .7 /min 1 /max 50" in one line ===
    def apply_param_change(kind: str, val_s: str | None) -> None:
        """
        Return: None. Applies a single param change (or toggle) in-place, printing the outcome.
        Logic: Parses kind/value, clamps where needed, updates globals (min/max/temp/p/k or r/rh/stop).
        Allowances: Accepts floats like '.7'; /r, /rh, /stop take no value; whitespace is allowed.
        """
        global MIN_NEW_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, STOP_EXTENSION

        if kind in ("r", "rh", "stop"):
            if kind == "r":
                randomize()
                print("\033[1;96m[Randomized Parameters]\033[0m")
            elif kind == "rh":
                randomize_high_variance()
                print("\033[1;96m[Randomized Parameters: High Variance]\033[0m")
            else:
                STOP_EXTENSION = not STOP_EXTENSION
                print(f"\033[1;94m[Stop extension {'ENABLED' if STOP_EXTENSION else 'DISABLED'}]\033[0m")
            return

        if val_s is None or not val_s.strip():
            # No numeric given → just show current params
            show_params()
            return

        # Normalize numbers like ".7" to "0.7"
        if val_s.startswith("."):
            val_s = "0" + val_s

        try:
            num = float(val_s)
        except ValueError:
            print("\033[1;91m[Error] Invalid number.\033[0m")
            return

        if kind == "min":
            MIN_NEW_TOKENS = max(0, int(num))
            if MAX_NEW_TOKENS < MIN_NEW_TOKENS:
                MAX_NEW_TOKENS = MIN_NEW_TOKENS
            print(f"\033[1;96m[min → {MIN_NEW_TOKENS}]\033[0m")

        elif kind == "max":
            MAX_NEW_TOKENS = max(1, int(num))
            if MIN_NEW_TOKENS > MAX_NEW_TOKENS:
                MIN_NEW_TOKENS = MAX_NEW_TOKENS
            print(f"\033[1;96m[max → {MAX_NEW_TOKENS}]\033[0m")

        elif kind in ("temp", "t"):
            TEMPERATURE = max(0.01, min(float(num), 2.0))
            print(f"\033[1;96m[temp → {TEMPERATURE:.2f}]\033[0m")

        elif kind == "p":
            TOP_P = max(0.05, min(float(num), 1.0))
            print(f"\033[1;96m[p → {TOP_P:.2f}]\033[0m")

        elif kind == "k":
            TOP_K = max(0, int(num))
            print(f"\033[1;96m[k → {TOP_K}]\033[0m")


    def parse_and_apply_multi_params(cmd: str) -> bool:
        """
        Return: True if at least one /param token was found and applied, else False.
        Logic: Scans the whole line for repeated tokens /(min|max|temp|t|p|k|r|rh|stop) with optional numbers; applies in order.
        Allowances: Free spacing; supports floats like '.7'; mixed toggles (/r /rh /stop) with numeric params in one command.
        """
        # find all occurrences anywhere in the line
        pattern = re.compile(
            r'/(min|max|temp|t|p|k|r|rh|stop)(?:\s+([+-]?(?:\d+(?:\.\d+)?|\.\d+)))?',
            flags=re.IGNORECASE
        )
        found = False
        for kind, val in pattern.findall(cmd):
            found = True
            apply_param_change(kind.lower(), val if val else None)
        return found


    if parse_and_apply_multi_params(cmd):
        generation_params = get_generation_params()
        continue

    limited_history_for_prompt = chat_history[-cached_history_size:]

    if history_enabled and cot_max_exchanges > 0:
        limited_history = limited_history_for_prompt[-cot_max_exchanges:]
        if DEEPHERMES:
            context = "\n".join(
                f"<|start_header_id|>user<|end_header_id|>\n{u}\n<|start_header_id|>assistant<|end_header_id|>\n{a}"
                for u, a in limited_history
            ) + "\n"
        else:
            context = "\n".join(
            f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
            for u, a in limited_history
        ) + "\n"
    else:
        context = ""

    if user_input.startswith("<|im_start|>"):
        current_prompt = f"{user_input}\n"
    elif DEEPHERMES:
        current_prompt = f"{SYSTEM_PROMPT}{context}<|start_header_id|>user<|end_header_id|>\n{user_input}\n<|start_header_id|>assistant<|end_header_id|>\n"
    else:
        if turn == 0:
            current_prompt = f"{SYSTEM_PROMPT}{context}<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        else:
            current_prompt = f"{SYSTEM_PROMPT}\n{context}<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    if BLANK:
        current_prompt = user_input
    
    if JUST_SYSTEM_PROMPT:
        if DEEPHERMES:
            current_prompt = f"{SYSTEM_PROMPT}{user_input}"
        else:
            current_prompt = f"{SYSTEM_PROMPT}\n{user_input}"

    if NO_SYSTEM_PROMPT:
        if DEEPHERMES:
            current_prompt = f"{context}<|start_header_id|>user<|end_header_id|>\n{user_input}\n<|start_header_id|>assistant<|end_header_id|>\n"
        else:
            current_prompt = f"{context}<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    if NO_ASSISTANT_PROMPT:
        if DEEPHERMES:
            current_prompt = f"{SYSTEM_PROMPT}{context}<|start_header_id|>user<|end_header_id|>\n{user_input}"
        else:
            current_prompt = f"{SYSTEM_PROMPT}\n{context}<|im_start|>user\n{user_input}"

    while count_tokens(current_prompt) > 3000 and chat_history:
        chat_history.pop(0)
        limited_history_for_prompt = chat_history[-cached_history_size:]
        if history_enabled and cot_max_exchanges > 0:
            limited_history = limited_history_for_prompt[-cot_max_exchanges:]
            context = "\n".join(
                f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
                for u, a in limited_history
            ) + "\n"
        else:
            context = ""
        current_prompt = f"{SYSTEM_PROMPT}\n{context}<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant"

    if CODE_CHECK:
        retry_count = 0
        reply = ""

        while retry_count < max_retries:
            if GGUF:
                output_stream = model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    max_tokens=MAX_NEW_TOKENS,
                    stream=True
                )

                text = ""
                for chunk in output_stream:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                    text += delta
            else:
                inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
                streamer = CyanPromptStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        **generation_params,
                        streamer=streamer
                    )
                text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

            reply = extract_assistant_reply(text)
            stripedreply = strip_code_word(reply)
            text = strip_begin_of_text(text)

            if not BLANK:
                is_code, scores = classify(stripedreply)
                if is_code:
                    retry_count += 1
                    log(f"\033[1;91mClassifier scores: code={scores.get('code', 0):.3f}, chat={scores.get('chat', 0):.3f}\033[0m")
                    log(f"\033[1;91mAttempt {retry_count} reply:\n{text}\n\033[0m")
                    log(f"\033[1;91mAttempt {retry_count} stripped reply:\n{reply}\n\033[0m")
                    log(f"\033[1;92;102m[Trying to code. Reprompting... {retry_count}/{max_retries}]\033[0m")
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(f"[Trying to code. Reprompting... {retry_count}/{max_retries}]\n")
                else:
                    break

        final_check, final_scores = classify(text)
        if retry_count == max_retries and final_check:
            log(f"\033[1;91mFinal classifier scores: code={final_scores.get('code', 0):.3f}, chat={final_scores.get('chat', 0):.3f}\033[0m")
            polite_message = "Sorry, I’m not able to provide code snippets."
            log(polite_message)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"{polite_message}\n")
            if history_enabled:
                chat_history.append((user_input, polite_message))
    else:
        if GGUF:
            output_stream = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_NEW_TOKENS,
                stream=True  # Enable streaming
            )

            text = ""
            for chunk in output_stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)  # Optional: stream live to console
                text += delta
        else:
            inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
            streamer = CyanPromptStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            with torch.no_grad():
                try:
                    print("\033[1;35m> \033[0m ", end="", flush=True)
                    output_ids = generate_prefer_short_then_extend(
                        model, tokenizer,
                        **inputs,
                        **generation_params,
                        first_min=MIN_NEW_TOKENS,
                        first_max=MAX_NEW_TOKENS,
                        extend_step=64,
                        hard_cap=1024,
                        streamer=streamer
                    )
                except KeyboardInterrupt:
                    streamer.stopped = True
                    print("\n\033[1;91m[Generation interrupted by Ctrl+C — using partial output.]\033[0m")

            text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            
        originalreply = extract_assistant_reply(text)
        text = lowercase_lines_and_sentences(text)
        reply = lowercase_lines_and_sentences(originalreply)
        stripedreply = strip_code_word(reply)
        text = strip_begin_of_text(text)
        logtext = text
        text = colorize(text)
        consoletext = clean_special_tokens(text)
        #text = filter_emojis_keep_first_single(text)
        #text = remove_all_emojis(text)
        print(f"\n{consoletext}")
        justlog(logtext)
        if history_enabled:
            if not GGUF:
                chat_history.append((user_input, reply if reply.strip() else streamer.output.strip()))
            else:
                chat_history.append((user_input, reply.strip()))
            turn += 1
            if args.auto and not auto_inputs:
                print("\n\033[1;92m[Auto mode finished — exiting.]\033[0m")
                break