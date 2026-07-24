"""Load the fine-tuned adapter on top of base Qwen2.5-1.5B and parse log lines.

Usage:
    python inference.py                      # runs the built-in examples
    python inference.py "your log line here" # parses a single line

The adapter is a ~189 MB diff, not a full model. This script loads the base
model from Hugging Face and applies the LoRA adapter on top of it.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER    = "adapter"   # local folder, or a HF Hub repo id like "saihrudhay/qwen1.5b-logparse-lora"

SYSTEM = """You are a log parser. Convert the security log line into a single JSON object.

Use exactly one of these event schemas:
- failed_login (ssh): event, service, user, src_ip, port
- successful_login (ssh): event, service, user, src_ip, port
- privilege_escalation (sudo): event, service, user, command
- http_request (nginx): event, service, src_ip, method, path, status
- firewall_block (ufw): event, service, src_ip, dst_ip, dst_port
- failed_login (windows): event, service, user, src_ip, logon_type

Ports, status codes and logon types are integers. Output only the JSON object, nothing else."""

EXAMPLES = [
    "Nov 02 14:20:11 web-01 sshd[3921]: Failed password for root from 8.8.8.8 port 5000 ssh2",
    '198.51.100.9 - - [02/Nov/2026:14:21:03 +0530] "POST /admin/login HTTP/1.1" 403 512 "-" "Mozilla/5.0"',
    "Nov 02 14:22:41 db-prod-2 sudo: dev01 : TTY=pts/1 ; PWD=/home/dev01 ; USER=root ; COMMAND=/bin/cat /etc/shadow",
]


def load():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return model, tok


def ask(model, tok, log_line):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": log_line}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    new = out[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    model, tok = load()
    lines = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EXAMPLES
    for line in lines:
        print("\nLOG :", line)
        print("JSON:", ask(model, tok, line))
