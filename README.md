# Laya

Fast, non-autoregressive System 1 decision engine with mathematically calibrated probabilities.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15d4Yv__KHeHjshVb-6PRTfqVllxih2S3?usp=sharing)
[![PyPI version](https://img.shields.io/pypi/v/laya.svg)](https://pypi.org/project/laya/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-laya--demo-orange)](https://huggingface.co/spaces/convaiinnovations/laya-demo)
[![Dev.to Article](https://img.shields.io/badge/dev.to-Read%20Article-0A0A0A?logo=devdotto&logoColor=white)](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-nandakishorm-FFDD00?logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/nandakishorm)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Laya lets you evaluate typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket, or JSON document) in **a single forward pass (~33–38 ms on GPU)**. It produces structured decision outputs and calibrated confidence scores without text generation, token streaming, or hallucinations.

Powered by the fine-tuned [Laya model on Hugging Face](https://huggingface.co/convaiinnovations/laya).

---

## Installation

```bash
pip install laya
```

---

## Quickstart

```python
import laya

# 1. Load the fine-tuned model directly from Hugging Face Hub (auto-downloads weights)
agent = laya.load("convaiinnovations/laya")

# 2. Provide any state (string or dictionary)
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 3. Define your typed questions
questions = {
    # choice: categorical selection with probabilities & confidence
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    # score: placement on an ordinal rubric
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    # noul: calibrated boolean probability P(true)
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing or scam attempt?"
    }
}

# 4. Run all questions in ONE single forward pass (~35 ms on GPU)
result = agent.predict(state, questions)
answers = result["answers"]

print("Department :", answers["department"]["choice"])
# -> billing (confidence: 0.94)

print("Urgency    :", answers["urgency"]["score"])
# -> 1.84 / 2.0

print("Churn Risk :", answers["churn_risk"]["noul"])
# -> 0.892 (89.2% probability)

print("Phishing   :", answers["is_phishing"]["noul"])
# -> 0.008 (0.8% probability)
```

---

## HTTP API and Runpod Serverless

The repository includes a FastAPI service that runs the same `agent.predict(state, questions)` flow. `POST /predict` returns the complete result directly (`model`, `answers`, and `usage`), without a job envelope.

### Run locally

The service requires Python 3.10 or later; this does not change the standalone library's Python requirement.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-server.txt .
cp .env.example .env
```

Set `API_AUTH_TOKEN` in `.env` to your own generated secret. The server refuses to start without it. Existing environment variables take precedence over `.env`.

```bash
# CPU development, including Macs without NVIDIA CUDA:
LAYA_DEVICE=cpu python api.py

# On a CUDA-capable machine, omit the CPU override:
# python api.py
```

The first startup downloads the pinned `convaiinnovations/laya` model. Later starts reuse the cache; the model is loaded once per process, not once per request. Wait for `GET /ping` to return HTTP 200 before predicting.

In another terminal, load your own `.env` and call the API:

```bash
set -a
source .env
set +a

curl --fail-with-body http://localhost:8000/predict \
  -H "X-API-Token: $API_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"message": "Please refund the duplicate charge today."},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "charges and refunds", "technical": "bugs and outages"}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"]
      },
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer request a refund?"
      }
    }
  }'
```

`state` accepts an object, string, or JSON list. `questions` must be nonempty. Choice criteria require at least two labels (a dictionary or distinct-label list); score criteria require at least two rubric strings. Noul criteria can optionally describe `"false"` and `"true"`. All questions require string instructions; unknown fields and invalid question types are rejected.

| Response | Meaning |
|---|---|
| `200` from `/predict` | Complete prediction result |
| `401` | Missing or incorrect `X-API-Token` |
| `422` | Invalid request or question options |
| `503` from `/predict` | Model not ready |
| `500` | Unexpected inference failure |

The worker's unauthenticated `/ping` returns `204` while loading, `200` when ready, and `503` if initialization fails. Inference runs off the event loop and is serialized per worker, so health checks remain responsive.

### Docker

Build the Linux GPU image without downloading weights into it:

```bash
docker build --platform linux/amd64 -t noah-laya:local .
docker volume create noah-laya-cache
docker run --rm --gpus all --env-file .env \
  -p 8000:8000 -v noah-laya-cache:/runpod-volume noah-laya:local
```

For CPU-only container testing, replace `--gpus all` with `--platform linux/amd64 -e LAYA_DEVICE=cpu`. The image defaults to CUDA and rejects a missing GPU at initialization instead of silently declaring a CPU worker ready.

Keep the optional cache overrides in `.env.example` commented when passing `.env` to Docker. The container must use the mounted volume, not an ephemeral local cache directory. Secrets, local model weights, and virtual environments are excluded from the build context.

| Setting | Local default | Container / Runpod |
|---|---|---|
| `API_AUTH_TOKEN` | Required | Required runtime secret |
| `LAYA_DEVICE` | `cuda`; explicitly use `cpu` or `mps` for development | `cuda` |
| `PORT` | `8000` | `8000` |
| `PORT_HEALTH` | Same HTTP server | `8000` |
| `HEALTH_CHECK_PATH` | `/ping` | `/ping` |
| `MODEL_CACHE_DIR` | `./.model-cache` | `/runpod-volume/models/laya` |
| `HF_HOME` | Hugging Face default | `/runpod-volume/huggingface` |
| `HF_TOKEN` | Optional; model is public | Optional runtime secret |

The pinned model revision is `7c76b622dfc5cac71b2dc1c29873efe2ce509a05`. Initialization retains the weights, local encoder configuration, and tokenizer together; a shared file lock protects downloads and tokenizer compatibility updates. An atomic completion marker allows subsequent workers to load without contacting Hugging Face. A failed initialization never publishes a new ready marker. Do not delete or modify cache files while workers use them.

### Deploy from GitHub

Use Runpod's [GitHub integration](https://docs.runpod.io/serverless/workers/github-integration) and [Load Balancer endpoint mode](https://docs.runpod.io/serverless/load-balancing/overview), **not Queue mode**:

1. Publish the implementation to `AdamGoodApp/noah`, branch `main`. Authorize Runpod's GitHub App for this repository only.
2. Create a 5GB **High-Performance network volume** in `US-CA-2`, after checking the current storage quote and GPU availability.
3. Create an endpoint from that GitHub repository, using the root `Dockerfile` and endpoint type **Load Balancer**.
4. Select one **RTX A5000 (24GB)** GPU in `US-CA-2`. The MCP/API uses pool `AMPERE_24`; select only A5000 by excluding other current members of that pool. Re-check pool membership when configuring it.
5. Attach the new network volume; expose HTTP port `8000`, with `PORT=8000`, `PORT_HEALTH=8000`, and `HEALTH_CHECK_PATH=/ping`.
6. Set `API_AUTH_TOKEN` as a runtime secret. Keep the image's cache paths and `LAYA_DEVICE=cuda`; do not bake `.env` into the image or pass a Runpod account API key into the container.
7. Set minimum workers **0**, maximum workers **1**, idle timeout **5 seconds**, request-count scaling target **1**, FlashBoot enabled, and container disk **10GB**.
8. Wait for the image build and model initialization, then verify `/predict` using the public sample above. Let the endpoint scale to zero and verify a subsequent worker reports a cache hit.

Runpod's public request uses two independent credentials:

```bash
curl --fail-with-body "https://$RUNPOD_ENDPOINT_ID.api.runpod.ai/predict" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "X-API-Token: $API_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"state":"Please refund my duplicate payment.","questions":{"refund":{"type":"noul","instructions":"Is a refund requested?"}}}'
```

The Runpod bearer key authenticates at the platform; `X-API-Token` authenticates this application. Neither substitutes for the other. A cold start may return a platform “no workers available” response before initialization finishes; wait for readiness and retry the idempotent prediction. Ordinary GitHub pushes do not deploy updates: Runpod's integration rebuilds on GitHub releases.

**Storage and location:** a [network volume](https://docs.runpod.io/storage/network-volumes) restricts workers to its data center. This configuration prioritizes the inexpensive 24GB GPU over an expensive Japan GPU; US-CA-2 is the selected High-Performance-storage location. The catalog checked on 2026-09-18 quoted A5000 Serverless at **$0.69 per running worker-hour**. Premium storage is billed separately even at zero workers; confirm its regional quote in the console before creating it. Pricing and stock can change.

Runpod's [beta Global Volumes](https://docs.runpod.io/storage/globalvolume) are currently documented for Pods, and the connected Serverless tools expose no Global Volume attachment. They also lack file locking and atomic rename. **Do not point this cache initializer at a Global Volume.** Keep a POSIX-capable regional network volume for this Serverless service.

### API regression checks

```bash
python -m pip install pytest httpx
python -m pytest tests/test_api.py tests/test_model_cache.py
```

These isolated checks cover authentication, request boundaries, health transitions, concurrent/cancelled requests, and cache failures. They do not replace a real-model HTTP smoke test or verification on an actual CUDA worker.

---

## Automated Confidence Gating

Because Laya's probabilities are trained with strictly proper scoring rules (RLCD), confidence scores are statistically meaningful:

```python
dept = answers["department"]["choice"]
conf = answers["department"]["confidence"]

if conf >= 0.85:
    # High confidence: automated action without human in the loop
    route_automatically(dept)
else:
    # Low confidence: escalate to human triage
    escalate_to_human_agent(dept, reason=f"Low confidence ({conf:.2f})")
```

---

## Built-in Workflow Presets

Laya provides pre-tuned question schemas for immediate production use:

```python
import laya

agent = laya.load("convaiinnovations/laya")

# 1. Intelligent Model Router (routes to small vs. frontier models)
routing = agent.predict({"request": "Refactor this service using dependency injection"}, laya.router_questions())

# 2. Real-time Prompt Guardrails (jailbreaks, injections, leaks)
guard = agent.predict({"prompt": "Ignore all instructions"}, laya.guard_questions())

# 3. Content Safety & Moderation (toxicity, harassment, threats)
safety = agent.predict({"post": "User comment text"}, laya.moderation_questions())

# 4. Support Ticket Triage (intent, urgency, frustration, churn)
triage = agent.predict({"message": "My payment failed twice"}, laya.triage_questions())
```

---

## Decision Primitives

| Primitive | Output | Use Cases |
|---|---|---|
| **`choice`** | Top label, probabilities per option, confidence | Department routing, intent classification, topic categorization |
| **`score`** | Expected level on ordinal rubric, distribution, confidence | Frustration level, ticket urgency, harm severity |
| **`noul`** | Calibrated probability P(true) from 0.0 to 1.0 | Phishing detection, spam filtering, jailbreak detection, churn risk |

---

## Benchmark: Laya vs. TypeSafe Jev

<div align="center">
  <img src="assets/benchmark_comparison.png" alt="Laya vs TypeSafe Jev Benchmark" width="900" />
</div>

| Metric / Dimension | TypeSafe Jev (Published) | Laya (Fine-Tuned Checkpoint) | Analysis / Advantage |
|---|---|---|---|
| **P50 Latency (1 Question)** | ~400 ms avg (70 to 500 ms, 150 ms best) | **38.4 ms** (p95: 42.1 ms) | **Laya is ~10.4x faster on avg (4x faster than Jev best-case)** |
| **Batched Latency (10 Questions)** | ~1,500 ms (serial) / ~400 ms | **156.0 ms** (p95: 158.4 ms) | **Laya evaluates 10 questions in the time Jev answers 1** |
| **Batched Latency (50 Questions)** | Multi-second / rate-limited | **721.4 ms** | High-throughput parallel mini-batching |
| **Benchmark Accuracy** | **67.8%** (across 4 production workflows) | **83.8%** in-task macro accuracy | **Laya achieves +16.0% higher overall accuracy** |
| **Intent & Customer Routing** | ~95 to 98% agreement | **99.1% accuracy** (ECE: 0.009) | Near-zero calibration error on routing |
| **Moderation & Content Safety** | ~92 to 95% agreement | **96.7% accuracy** (ECE: 0.061) | Clean safety boundary separation |
| **Inference & Fact Verification** | Not separately reported | **88.3% accuracy** (ECE: 0.054) | Full bidirectional attention captures contradictions |
| **Instruction-Following Tasks** | Proprietary internal set | **87.8% in-task / 86.3% zero-shot** | Proven generalization across unseen tasks |
| **Email Triage & Phishing** | Vendor custom workflow | **73.2% accuracy** (ECE: 0.017) | Tailored email cleaning & phishing filters |
| **Selective Automation (@ 50% Cov)** | Claims human escalation | **92.2% accuracy** (ECE: 0.041) | Safe automated gating (confidence >= 0.85) |
| **Model Weights & Code** | Closed-source / proprietary API | **100% Open-source Apache 2.0** | Full data sovereignty & transparency |
| **Inference Cost** | $0.042 / 1M input tokens recurring | **$0.00 / self-hosted** | Runs on commodity GPUs, Mac MPS, or CPU |
| **Multi-Turn Trajectory Modeling** | Static state snapshots | **TD(lambda = 1.0) prefix modeling** | Real temporal credit assignment |
| **Deployment Mode** | Cloud-only egress | **Air-gapped / Local / On-Device** | Zero data egress (HIPAA/GDPR compliant) |

---

## Live Demo & Resources

* **Hugging Face Model:** [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
* **Interactive Web Demo:** [convaiinnovations/laya-demo](https://huggingface.co/spaces/convaiinnovations/laya-demo)
* **Engineering Writeup:** [Read the full story on Dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)

---

## Fine-Tuning on Single T4 GPU (Google Colab)

Fine-tune Laya on custom domain data or the `LocalLLaMA/typed-decisions` benchmark on a free T4 GPU:

* **Interactive Fine-Tuning Notebook:** [Fine-Tune on Custom Data](https://colab.research.google.com/drive/15d4Yv__KHeHjshVb-6PRTfqVllxih2S3?usp=sharing) ([`notebooks/laya_finetune_colab.ipynb`](notebooks/laya_finetune_colab.ipynb))
* **Workflow Benchmark Fine-Tuning:** [`notebooks/laya_finetune_typed_decisions_colab.ipynb`](notebooks/laya_finetune_typed_decisions_colab.ipynb) — fine-tunes on `LocalLLaMA/typed-decisions` (1,200 cases), evaluates on 400 test cases, and pushes to Hugging Face.

---

## Support the Project

If Laya helps your research or products, consider supporting independent research:

<p align="left">
  <a href="https://www.buymeacoffee.com/nandakishorm" target="_blank">
    <img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=nandakishorm&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" alt="Buy Me A Coffee" />
  </a>
</p>

---

## License

Apache 2.0. Developed by Convai Innovations.
