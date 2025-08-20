"""
interface.py - Discord-Micae Model Chat Interface

Description:
    This script provides an interactive CLI for chatting with Hugging Face language models,
    designed for Discord-style conversational flow. It supports both Transformers and GGUF
    (llama.cpp) models, with optional LoRA adapter loading and bitsandbytes quantization.

Key Features:
    • Interactive multi-line prompt entry with `prompt_toolkit`
    • Chain-of-thought context management with enable/disable commands
    • On-the-fly generation parameter tuning (/min, /max, /temp, /p, /k)
    • Rich-colored streaming token output
    • Emoji filtering and cleanup
    • Configurable prompt modes (system prompt, assistant prompt, blank mode)
    • Short-preference generation with extension until EOS
    • Optional code detection and filtering

Arguments:
    -c, --cot                  Disable chain-of-thought message context (default: enabled)
    -m, --model                Model path or Hugging Face repo ID (default: mookiezi/Discord-Micae-8B-Preview)
    --deephermes               Enable DeepHermes formatting instead of ChatML (default: False)
    --gguf                     Use GGUF model format with llama.cpp backend (default: False)
    --gguf-chat-format         Chat format for GGUF models (default: "chatml")
    --blank                    Raw user input only, no prompts/system context (default: False)
    --assistant-system-combo   Include both system and assistant system prompts (default: False)
    --assistant-system         Use assistant system prompt instead of standard (default: False)
    --just-system-prompt       Use only the system prompt with user input (default: False)
    --no-system-prompt         Do not include system prompt (default: False)
    --no-assistant-prompt      Do not include assistant prompt (default: False)
    --code-check               Enable code detection and filtering via classifier (default: False)
    --quantization             Enable bitsandbytes quantization (default: True)
    --bnb-4bit                 Load model in 4-bit mode (default: True)
    --bnb-8bit                 Load model in 8-bit mode (default: False)
    --custom-tokens            Add extra special tokens to tokenizer (default: False)

Usage:
    python interface.py -m mookiezi/Discord-Micae-8B-Preview
    python interface.py --gguf --gguf-chat-format alpaca
    python interface.py --blank
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

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from huggingface_hub import login
import argparse
import logging
import re
from transformers import TextStreamer
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
import gc
from huggingface_hub import login
import random
from transformers import LogitsProcessor
from transformers import LogitsProcessorList
import signal
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
import argparse

parser = argparse.ArgumentParser(description="HuggingFace Model Chat Interface")

parser.add_argument("-c", "--cot", action="store_true",
                    help="Disable chain-of-thought message context (default: enabled)")
parser.add_argument("-m", "--model", default="mookiezi/Discord-Micae-8B-Preview",
                    help="Model path or Hugging Face repo ID")
parser.add_argument("--deephermes", action="store_true", default=False,
                    help="Enable DeepHermes formatting instead of ChatML (default: False)")
parser.add_argument("--gguf", action="store_true", default=False,
                    help="Use GGUF model format (llama.cpp backend) (default: False)")
parser.add_argument("--gguf-chat-format", default="chatml",
                    help='Chat format for GGUF models (default: "chatml")')
parser.add_argument("--blank", action="store_true", default=False,
                    help="Use only raw user input (no prompts/system context) (default: False)")
parser.add_argument("--assistant-system-combo", action="store_true", default=False,
                    help="Include both system and assistant system prompts (default: False)")
parser.add_argument("--assistant-system", action="store_true", default=False,
                    help="Use assistant system prompt instead of standard system prompt (default: False)")
parser.add_argument("--just-system-prompt", action="store_true", default=False,
                    help="Use only the system prompt with user input (default: False)")
parser.add_argument("--no-system-prompt", action="store_true", default=False,
                    help="Do not include system prompt (default: False)")
parser.add_argument("--no-assistant-prompt", action="store_true", default=False,
                    help="Do not include assistant prompt (default: False)")
parser.add_argument("--code-check", action="store_true", default=False,
                    help="Enable code detection and filtering via classifier (default: False)")
parser.add_argument("--quantization", action="store_true", default=True,
                    help="Enable bitsandbytes quantization (default: True)")
parser.add_argument("--bnb-4bit", action="store_true", default=True,
                    help="Load model in 4-bit mode (default: True)")
parser.add_argument("--bnb-8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode (default: False)")
parser.add_argument("--custom-tokens", action="store_true", default=False,
                    help="Add extra special tokens to tokenizer (default: False)")

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
QUANTIZATION = args.quantization
BNB_4BIT = args.bnb_4bit
BNB_8BIT = args.bnb_8bit
CUSTOM_TOKENS = args.custom_tokens
# ================================
base_model_name = args.model
FROZEN_LORA_PATH = None
checkpoint_path = ""
checkpoint_subfolder = None
# ================================
PLAIN_SYSTEM_PROMPT = """you are a person."""
ASSISTANT_SYSTEM_PROMPT = """"""
# ================================
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = random.randint(40, 75)
TEMPERATURE = random.uniform(0.5, 0.9)
TOP_P = random.uniform(0.7, 0.9)
TOP_K = random.randint(40, 75)
# ================================

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
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, input_ids, scores):
        # Get the last `window_size` tokens
        recent_tokens = input_ids[0, -self.window_size:].tolist()

        # Penalize their logits heavily
        for token_id in set(recent_tokens):
            scores[0, token_id] = -float("inf")

        return scores
    
processor_list = LogitsProcessorList()
processor_list.append(RecentTokenBlocker(window_size=3))

def get_generation_params(current_prompt, user_input):
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
        "min_p": .08,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.eos_token_id if tokenizer else None,
        "logits_processor": processor_list
    }

logging.getLogger("transformers").setLevel(logging.ERROR)

bnb_config4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

bnb_config8bit = BitsAndBytesConfig(
    load_in_8bit=True 
)

if BNB_4BIT:
    bnb_config = bnb_config4bit

if BNB_8BIT:
    bnbconfig = bnb_config8bit


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
        n_gpu_layers=-1,
        n_ctx=4096,
        chat_format=GGUF_CHAT_FORMAT
    )
    tokenizer = None
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if CUSTOM_TOKENS:
        special_tokens = {
            "additional_special_tokens": [
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.eos_token = "<|eot_id|>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if USE_BASE_MODEL_ONLY:
        if QUANTIZATION:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                device_map="auto",
                #low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                #low_cpu_mem_usage=True
            )
        model.resize_token_embeddings(len(tokenizer))
    else:
        # Load base model first
        if QUANTIZATION:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                device_map="auto",
                #low_cpu_mem_usage=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                #low_cpu_mem_usage=True
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

    def on_text(self, text, **kwargs):
        text = strip_begin_of_text(text)
        text = clean_special_tokens(text)
        #text = filter_emojis_keep_first_single(text)
        #text = remove_all_emojis(text)

        self.output += text  # capture output

        if self.first:
            print("\033[1;96m> \033[0m", end='', flush=True)
            self.first = False
        print(text, end='', flush=True)

    def end(self):
        print()
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

if not GGUF:
    model.eval()

if CODE_CHECK:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

folder_name = f"{base_model_name.replace('/', '_')}_{os.path.basename(checkpoint_path).replace('/', '_')}"
output_dir = os.path.join(os.getcwd(), "output", folder_name)
os.makedirs(output_dir, exist_ok=True)
base_filename = folder_name
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

chat_history = []

def log(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(colorize(text))
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {text}\n\n────────────────────────────────────────────────────────────────────────────\n\n")

def justlog(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {text}\n\n────────────────────────────────────────────────────────────────────────────\n\n")

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
        if not line:
            return ''
        first_word = line.split()[0]
        if should_skip(first_word):
            return line
        return line[:1].lower() + line[1:]
    
    lines = [lower_line_start(line) for line in text.splitlines()]
    text = "\n".join(lines)
    
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
generation_params = get_generation_params(current_prompt, user_input)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"# cot_enabled: {args.cot}\n")
    f.write(f"# generation_params: {generation_params}\n\n────────────────────────────────────────────────────────────────────────────\n\n")

max_retries = 5
cot_enabled = not args.cot
cot_max_exchanges = 1000
cached_history_size = 1000

if not GGUF:
    print(model.active_adapter)

print(f"🚀 Model launched.")

turn = 0

#=============
# START LOOP
#=============

while True:
    generation_params = get_generation_params(current_prompt, user_input)
    show_params()

    user_input = session.prompt(multiline=True)
    user_input = user_input.strip()

    cmd = user_input.lower()

   # --- clear/reset first (includes /c and !c aliases) ---
    if cmd in ["/clear", "!clear", "/reset", "!reset", "/c", "!c"]:
        chat_history.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print("\033[1;93m[Chain-of-thought history cleared.]\033[0m")
        log("[Chain-of-thought history cleared.]")
        continue

    # --- /back (/b): undo last exchange and reprint tail ---
    if cmd in ["/back", "!back", "/b", "!b"]:
        if chat_history:
            u, a = chat_history.pop()
            turn = max(0, turn - 1)
            print("\033[1;93m[Undo] Removed last user+assistant exchange.\033[0m")
            log("[Undo] Removed last user+assistant exchange.]")

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
                print("\033[1;90m[History is now empty.]\033[0m")
        else:
            print("\033[1;90m[Nothing to undo.]\033[0m")
        continue

    # --- /h (CoT on, optional count) ---
    m = re.match(r"^(/h|!h)(\s+(\d+))?$", cmd)
    if m:
        cot_enabled = True
        cot_max_exchanges = int(m.group(3)) if m.group(3) else len(chat_history)
        print(f"\033[1;94m[Chain-of-thought] Using last {cot_max_exchanges} exchanges.]\033[0m")
        log(f"[Chain-of-thought] Using last {cot_max_exchanges} exchanges.]")
        continue

    # --- /d (CoT off) ---
    if cmd in ["/d", "!d"]:
        cot_enabled = False
        cot_max_exchanges = 0
        print("\033[1;94m[Chain-of-thought disabled.]\033[0m")
        log("[Chain-of-thought disabled.]")
        continue


    # === /min /max /temp /t /p and /k for paramsmeter changing on the fly ===
    m = re.match(r"^/(min|max|temp|t|p|k)\s*([0-9]*\.?[0-9]+)?$", cmd)
    if m:
        kind = m.group(1)
        val_s = m.group(2)

        if val_s is None:
            show_params()
            continue

        try:
            val = int(float(val_s)) if kind in ("min", "max", "k") else float(val_s)
        except ValueError:
            print("\033[1;91m[Error] Invalid number.\033[0m")
            continue

        # clamp + assign
        if kind == "min":
            MIN_NEW_TOKENS = max(0, int(val))
            if MAX_NEW_TOKENS < MIN_NEW_TOKENS:
                MAX_NEW_TOKENS = MIN_NEW_TOKENS
            print(f"\033[1;96m[min → {MIN_NEW_TOKENS}]\033[0m")

        elif kind == "max":
            MAX_NEW_TOKENS = max(1, int(val))
            if MIN_NEW_TOKENS > MAX_NEW_TOKENS:
                MIN_NEW_TOKENS = MAX_NEW_TOKENS
            print(f"\033[1;96m[max → {MAX_NEW_TOKENS}]\033[0m")

        elif kind in ("temp", "t"):
            TEMPERATURE = max(0.01, min(float(val), 2.0))
            print(f"\033[1;96m[temp → {TEMPERATURE:.2f}]\033[0m")

        elif kind == "p":
            TOP_P = max(0.05, min(float(val), 1.0))
            print(f"\033[1;96m[p → {TOP_P:.2f}]\033[0m")

        elif kind == "k":
            TOP_K = max(0, int(val))
            print(f"\033[1;96m[k → {TOP_K}]\033[0m")

        generation_params = get_generation_params(current_prompt, user_input)
        show_params()
        continue

    if cmd in ("/params", "/settings", "/gen"):
        print(
            f"\033[1;92m[Params]"
            f" min={MIN_NEW_TOKENS} | max={MAX_NEW_TOKENS} |"
            f" temp={TEMPERATURE:.2f} | p={TOP_P:.2f} | k={TOP_K}\033[0m"
        )
        continue


    limited_history_for_prompt = chat_history[-cached_history_size:]

    if cot_enabled and cot_max_exchanges > 0:
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
        if cot_enabled and cot_max_exchanges > 0:
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
            if cot_enabled:
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
            streamer = CyanPromptStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.no_grad():
                try:
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
        print(consoletext)
        justlog(logtext)
        if cot_enabled:
            chat_history.append((user_input, reply if reply.strip() else streamer.output.strip()))
            turn += 1