# Discord-Micae Model Chat Interface

A Python-based interactive CLI interface for chatting with Hugging Face language models, optimized for casual, Discord-style conversation using ChatML.
Supports both quantized and full-precision models, live token streaming with color formatting, and dynamic generation parameter adjustment.

---

<p align="center">
  <img src="https://github.com/mookiezi/site/blob/main/interface-screenshot.png?raw=true" alt="Interface Screenshot">
</p>

---

## Features

-   **Multiple Model Formats**

    -   Hugging Face Transformers (`AutoModelForCausalLM`)
    -   GGUF (llama.cpp) backend
    -   LoRA adapter loading
    -   4-bit / 8-bit quantization with bitsandbytes

-   **Custom Prompt Controls**

    -   Chain-of-Thought context management
    -   Raw blank mode, no system prompts, or assistant-only modes
    -   DeepHermes and ChatML formatting options
    -   Optional code detection and filtering

-   **Interactive Chat**

    -   Multi-line input with `prompt_toolkit`
    -   Persistent conversation history (`/back`, `/clear`)
    -   Runtime parameter adjustment (`/min`, `/max`, `/temp`, `/p`, `/k`, `/r`, `/rh`)

-   **Streaming Output**
    -   Token-by-token display with Rich coloring
    -   Emoji filtering and cleanup
    -   Automatic lowercasing rules
    -   **EOS-Aware Extension:** starts with a short randomized budget (40–75 tokens), then automatically extends generation in steps (64 tokens) until `<|im_end|>` or EOS is reached, a hard cap (1024 tokens), or manual `/stop` is triggered

---

## Installation

Install with `requirements.txt`:

```
pip install -r requirements.txt
```

Or install manually:

```
pip install torch transformers peft bitsandbytes prompt_toolkit rich
```

## Optional dependencies

If using GGUF (llama.cpp models):

```
pip install llama-cpp-python
```

---

## CLI Arguments (with defaults)

```
usage: interface.py [-h] [-c] [-m MODEL]
                    [--deephermes] [--gguf] [--gguf-chat-format FORMAT]
                    [--blank] [--assistant-system-combo] [--assistant-system]
                    [--just-system-prompt] [--no-system-prompt]
                    [--no-assistant-prompt] [--code-check]
                    [--quantization] [--bnb-4bit] [--bnb-8bit]
                    [--custom-tokens]

optional arguments:
    -h, --help                Show this help message and exit
    -m MODEL, --model MODEL   Model path or Hugging Face repo ID
                            (default: mookiezi/Discord-Micae-8B-Preview)

Feature toggles (defaults in parentheses):
    -hs, --history                  History and message context (default: enabled)
    -fl, --frozen_lora              Model path or Hugging Face repo ID of the base LoRa adatper to load and freeze
    -c, --checkpoint                Model path or Hugging Face repo ID of the LorA adapter to load
    -chs, --checkpoint_subfolder    Subfolder of the path or Hugging Face repo ID of the LorA adapter to load")
    --deephermes                    Enable DeepHermes formatting instead of ChatML (default: False)
    --gguf                          Use GGUF model format with llama.cpp backend (default: False)
    --gguf-chat-format              Chat format for GGUF models (default: "chatml")
    --blank                         Raw user input only, no prompts/system context (default: False)
    -asc, --assistant-system-combo  Include both system and assistant system prompts (default: False)
    -as, --assistant-system         Use assistant system prompt instead of standard (default: False)
    --just-system-prompt            Use only the system prompt with user input (default: False)
    --no-system-prompt              Do not include system prompt (default: False)
    --no-assistant-prompt           Do not include assistant prompt (default: False)
    --code-check                    Enable code detection and filtering via classifier (default: False)
    --quantization                  Enable bitsandbytes quantization (default: True)
    --bnb-4bit                      Load model in 4-bit mode (default: True)
    --bnb-8bit                      Load model in 8-bit mode (default: False)
    --custom-tokens                 Add extra special tokens to tokenizer (default: False)
```

---

## Default Parameters

-   MIN_NEW_TOKENS = 1
-   MAX_NEW_TOKENS = `random.randint(40, 75)`
-   TEMPERATURE = `random.uniform(0.5, 0.9)`
-   TOP_P = `random.uniform(0.7, 0.9)`
-   TOP_K = `random.randint(40, 75)`
-   MIN_P = 0.08
-   NO_REPEAT_NGRAM_SIZE = 3
-   REPETITION_PENALTY = 1.2
-   EOS Handling = `<|im_end|>` and `tokenizer.eos_token_id` (extension continues until one is reached, or hard cap of 1024 tokens)

## Commands

| Command                 | Description                                                                 |
| ----------------------- | --------------------------------------------------------------------------- |
| `/clear` `/reset` `/c`  | Clear conversation history                                                  |
| `/back` `/b`            | Undo last user+assistant exchange and preview recent history                |
| `/h VAL`                | Enable Chain-of-Thought with last VAL exchanges (default: all available)    |
| `/d`                    | Disable Chain-of-Thought                                                    |
| `/min VAL`              | Set **min_new_tokens** to VALb                                              |
| `/max VAL`              | Set **max_new_tokens** to VAL                                               |
| `/temp VAL` or `/t VAL` | Set **temperature** to VAL                                                  |
| `/p VAL`                | Set **top_p** to VAL                                                        |
| `/k VAL`                | Set **top_k** to VAL                                                        |
| `/params` `/settings`   | Show current generation parameters                                          |
| `/r`                    | Randomize parameters (short-range defaults)                                 |
| `/rh`                   | Randomize parameters with **high variance** (wider temp/top_p/top_k ranges) |
| `/stop`                 | Toggle extension **ON/OFF** (controls continuation beyond initial budget)   |

---

## License

MIT License
