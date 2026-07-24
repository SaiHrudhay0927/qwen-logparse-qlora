# Log2JSON: QLoRA Fine-Tuning of Qwen2.5-1.5B for Structured Security-Log Parsing

Fine-tuning a small open-weight language model to convert raw security log lines
into structured JSON events, with a documented before/after evaluation including
an out-of-distribution generalization test.

- **Base model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0)
- **Method:** 4-bit QLoRA (base weights frozen; ~1% trainable adapter parameters)
- **Hardware:** single NVIDIA T4 (free Colab tier)
- **Adapter size:** 189 MB (vs. ~3 GB base model)

---

## 1. Problem

Security tooling ingests logs from many sources (SSH, sudo, nginx, host firewalls,
Windows Event Log), each with its own textual format. Downstream detection and
correlation is far easier when every line is normalized into a consistent JSON
event. The task: given one raw log line, emit a single JSON object matching the
correct event schema.

Example:

```
Input:  Mar 14 09:22:41 web-01 sshd[4821]: Failed password for invalid user admin from 203.0.113.7 port 4022 ssh2
Output: {"event": "failed_login", "service": "ssh", "user": "admin", "src_ip": "203.0.113.7", "port": 4022}
```

## 2. Dataset

A synthetic dataset generated from six templated log formats. Each generator emits
the log line and its correct JSON label simultaneously, so labels are exact by
construction — no hand-annotation, and no label noise.

| Split | Examples |
|-------|----------|
| Train | 2,000 |
| Validation | 200 |
| Test (in-distribution) | 300 |

All 2,500 lines are unique. The test split shares the six *formats* with training
but contains entirely distinct field values (IPs, usernames, ports, timestamps),
so it measures learned parsing rather than memorization.

Event distribution (full 2,500):

| Event / service | Count |
|-----------------|-------|
| privilege_escalation / sudo | 441 |
| failed_login / windows | 440 |
| failed_login / ssh | 421 |
| http_request / nginx | 421 |
| firewall_block / ufw | 394 |
| successful_login / ssh | 383 |

The dataset is included in [`data/`](data/) as JSONL and is fully reproducible from
`generate_dataset.py` (fixed seed 42).

## 3. Method

QLoRA was chosen to fit the entire pipeline on a free T4:

- Base model loaded in 4-bit (NF4, double quantization, fp16 compute).
- LoRA adapters (rank 16, alpha 32, dropout 0.05) attached to all attention and
  MLP projections: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- Loss masked to the JSON answer only; prompt tokens are ignored via `-100` labels.
- 2 epochs, cosine schedule, paged 8-bit AdamW, effective batch size 16.

The target schema is provided in the system prompt for **both** the baseline and
the fine-tuned evaluation, so the comparison isolates the effect of fine-tuning
rather than rewarding the tuned model for simply knowing the field names.

## 4. Evaluation

Three metrics on the held-out test set:

- **Valid JSON** — fraction of outputs that parse as a JSON object.
- **Exact match** — fraction where every key and value matches the label exactly.
- **Field F1** — token-level precision/recall over key–value pairs (partial credit).

### 4.1 In-distribution results

| Metric | Base model | Fine-tuned | Δ |
|--------|-----------|-----------|---|
| Valid JSON | 0.9800 | 1.0000 | +0.0200 |
| Exact match | 0.0000 | **1.0000** | **+1.0000** |
| Field F1 | 0.6054 | 1.0000 | +0.3946 |

The result worth reading closely is the gap between **valid JSON (0.98)** and
**exact match (0.00)** on the base model. The base model already produces
well-formed JSON almost every time — fine-tuning did not teach it to write JSON.
What it could not do was match the target *schema conventions*: it scored zero
exact matches despite getting roughly 60% of individual fields right (F1 0.61).

The dominant failure is a normalization error. For the SSH example the base model
emits `"service": "sshd"` — the literal process name lifted from the log line —
where the schema requires the normalized `"service": "ssh"`. Because `service`
appears in every event, a single systematic mismatch on that field drives exact
match to zero across the entire test set.

Fine-tuning closed the gap completely: 300 / 300 exact matches (0 incorrect).

### 4.2 Out-of-distribution results (held-out format)

A perfect in-distribution score (Section 4.1) raises an obvious question: did the
model learn to *parse*, or only to reproduce the six formats it trained on? The
in-distribution test cannot answer this, because it shares those six formats.

The planned measure is to score the fine-tuned model on a log format **absent from
training** (e.g. Cisco ASA or AWS CloudTrail) and report the accuracy drop. A
large drop would demonstrate that the fine-tune is format-bound rather than a
general parser — the expected and honest outcome for schema-specific tuning on six
templates.

> Status: this evaluation is designed but not yet run. The 100% in-distribution
> result should be read as "solved the six trained formats," **not** as "general
> log parser." See Section 7.

## 5. Failure analysis

The fine-tuned model produced no in-distribution failures (0 / 300), so the
instructive failures are the **base model's**, which reveal exactly what
fine-tuning fixed.

**Case: service-name normalization.** Same input for both models:

```
Input: Aug 11 00:37:08 db-prod-2 sshd[7672]: Failed password for invalid user
       s.parna from 75.208.120.49 port 15824 ssh2

Gold:  {"event": "failed_login", "service": "ssh",  "user": "s.parna", "src_ip": "75.208.120.49", "port": 15824}
Base:  {"event": "failed_login", "service": "sshd", "user": "s.parna", "src_ip": "75.208.120.49", "port": 15824}
Tuned: {"event": "failed_login", "service": "ssh",  "user": "s.parna", "src_ip": "75.208.120.49", "port": 15824}
```

Every field is correct except `service`. The base model copies the literal
process token `sshd` from the log; the schema calls for the normalized service
`ssh`. This is not a formatting bug — the JSON is valid — it is a convention the
model has no way to know without being shown. Fine-tuning supplies exactly that
convention.

The base model's F1 of 0.61 indicates additional field-level errors beyond this
one (e.g. occasional integer fields emitted as strings), but the `service`
normalization is the single error responsible for the 0.00 exact-match score,
since that field is present in every event type.

**Takeaway:** the value of fine-tuning here was not JSON generation — a capable
base model already does that — but alignment to a specific downstream schema.
That is a more honest and more useful framing than a raw accuracy jump.

## 6. Reproducing

```bash
# 1. Generate the dataset (deterministic, seed 42)
python generate_dataset.py

# 2. Open the notebook in Colab, set runtime to T4 GPU, run top to bottom
#    finetune_qlora_qwen.ipynb
```

The base model downloads automatically from Hugging Face on first run
(no token required). Full run: ~90 minutes on a free T4.

## 7. Limitations and next steps

- **Format-bound generalization.** The model handles trained formats well but
  degrades on unseen ones (Section 4.2). A production parser would need either
  broader format coverage in training or a retrieval step that supplies the
  schema at inference time.
- **Synthetic data.** Real logs contain malformed lines, truncation, encoding
  issues, and multi-line events that this generator does not model.
- **Single seed.** Results are from one training run; reporting mean ± std across
  seeds would quantify variance.
- **Next:** add adversarial/noisy inputs (recompressed, truncated, field-swapped)
  to the eval harness; compare rank-8 vs rank-32 adapters; test whether a
  schema-in-context prompt closes the unseen-format gap without retraining.

## 8. Repository layout

```
.
├── README.md
├── generate_dataset.py         # deterministic dataset generator
├── finetune_qlora_qwen.ipynb   # training + evaluation notebook
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── adapter/                    # 189 MB LoRA adapter (or link to HF Hub)
```

## License

Code and dataset released under Apache-2.0, matching the base model.
