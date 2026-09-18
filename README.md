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

## Runpod Queue Serverless

The repository includes a Queue worker that runs the same `agent.predict(state, questions)` flow. Runpod accepts `{"input": {"state": ..., "questions": ...}}` jobs and wraps the complete prediction (`model`, `answers`, and `usage`) in the job's `output`. Only Runpod's platform bearer key authenticates remote clients; the worker has no separate application credential or HTTP service.

### Run locally

The worker requires Python 3.10 or later; this does not change the standalone library's Python requirement.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-server.txt .
cp .env.example .env
```

The optional `.env` is for local configuration; existing environment variables take precedence. No API key is required for a local SDK test. Define a public synthetic job and run it through the [SDK's local test mode](https://docs.runpod.io/serverless/development/local-testing):

```bash
JOB_INPUT='{
  "input": {
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
  }
}'

# CPU development, including Macs without NVIDIA CUDA:
LAYA_DEVICE=cpu python handler.py --test_input "$JOB_INPUT"

# On a CUDA-capable machine:
# python handler.py --test_input "$JOB_INPUT"
```

The first startup downloads the pinned `convaiinnovations/laya` model. Later starts reuse the cache. The model loads once per process **before** the SDK starts accepting jobs; initialization failure stops the worker. Prediction is synchronous and serialized per worker. Do not enable concurrent-handler processing.

`input.state` accepts an object, string, or JSON list. `input.questions` must be nonempty. Choice criteria require at least two labels (a dictionary or distinct-label list); score criteria require at least two rubric strings. Noul criteria can optionally describe `"false"` and `"true"`. All questions require string instructions; unknown fields and invalid question types are rejected.

### Submit and retrieve jobs

After deploying the Queue endpoint, set `RUNPOD_ENDPOINT_ID` to its ID. **There is no new live Queue endpoint ID documented yet.** Set `RUNPOD_API_KEY` in your client environment or load your private local `.env`; never copy the client key into Runpod's worker environment.

```bash
# Client terminal only; .env must be a file you control.
set -a
source .env
set +a
export RUNPOD_ENDPOINT_ID="YOUR_QUEUE_ENDPOINT_ID"

# Asynchronous submission is preferable for a first request/cold start.
curl --fail-with-body "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$JOB_INPUT"
```

The response initially contains a job ID and status, for example `{"id":"<job-id>","status":"IN_QUEUE"}`. Save that ID and poll the **same job**, waiting several seconds between requests:

```bash
export JOB_ID="ID_RETURNED_BY_RUN"
curl --fail-with-body \
  "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

For a synchronous submission instead of `/run`:

```bash
curl --fail-with-body --max-time 310 \
  "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync?wait=300000" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$JOB_INPUT"
```

`/runsync` waits up to the requested interval (milliseconds; currently up to 300000). Cold starts include capacity wait, image startup, model download if uncached, and model loading. If a response is still `IN_QUEUE` or `IN_PROGRESS`, retain its ID and use `/status/{id}` rather than submitting duplicate work. The wait parameter is not an execution timeout or a result-retention setting.

A completed job has this shape. **Numbers below are illustrative, not measured predictions**; Runpod may also include timing/worker metadata:

```json
{
  "id": "<job-id>",
  "status": "COMPLETED",
  "output": {
    "model": "laya-rl-agent",
    "answers": {
      "department": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.95, "technical": 0.05},
        "confidence": 0.8,
        "action": {"act_probability": 0.9}
      },
      "urgency": {
        "type": "score",
        "score": 1.1,
        "legend": {"0": "not urgent", "1": "soon", "2": "critical"},
        "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
        "confidence": 0.6,
        "action": {"act_probability": 0.8}
      },
      "refund_requested": {
        "type": "noul",
        "noul": 0.95,
        "confidence": 0.95,
        "action": {"act_probability": 0.9}
      }
    },
    "usage": {"input_tokens": 120, "output_tokens": 0}
  }
}
```

Consumers must inspect `status`, not just the HTTP status code:

| Job status | Meaning |
|---|---|
| `IN_QUEUE` | Waiting for an initialized worker |
| `IN_PROGRESS` | Worker is processing the job |
| `COMPLETED` | Read the full prediction from `output` |
| `FAILED` | Invalid input, unsupported question options, or inference failure; inspect the sanitized job error |
| `TIMED_OUT` / `CANCELLED` | No successful prediction; handle as a terminal failure |

Validation and inference errors are raised as sanitized exceptions so Runpod marks the job **FAILED**, not `COMPLETED` with an error dictionary. The worker does not expose original input or original exception messages through those errors. Platform authentication errors are separate: a missing/invalid bearer key is rejected before the worker handles a job.

Results are **not permanent storage**. Runpod currently documents `/run` results retained for **30 minutes after completion**, and `/runsync` results for **1 minute**. Retrieve and save needed results promptly; job TTL also limits overall lifetime, including queue time. See the [request lifecycle](https://docs.runpod.io/serverless/endpoints/send-requests) and [operation reference](https://docs.runpod.io/serverless/endpoints/operation-reference) for current limits.

### Docker and runtime configuration

Build the Linux GPU image without downloading weights into it. Reuse `JOB_INPUT` from the local example for this one-shot SDK test:

```bash
docker build --platform linux/amd64 -t noah-laya:local .
docker volume create noah-laya-cache
docker run --rm --gpus all \
  -v noah-laya-cache:/runpod-volume noah-laya:local \
  python handler.py --test_input "$JOB_INPUT"
```

For CPU-only container testing, replace `--gpus all` with `--platform linux/amd64 -e LAYA_DEVICE=cpu`. This does not verify CUDA. The image defaults to CUDA and rejects a missing GPU at initialization instead of silently declaring a CPU worker ready.

The deployed image's default command is `python handler.py`; Runpod supplies the Queue worker environment. No HTTP port or application health route needs exposing. **Do not pass the client `.env` using `--env-file`** or configure `RUNPOD_API_KEY` as a worker secret. Secrets, local model weights, and virtual environments are excluded from the build context.

| Setting | Local default | Container / Runpod |
|---|---|---|
| `RUNPOD_API_KEY` | Client-only; not needed for local SDK tests | Never pass to worker |
| `LAYA_DEVICE` | `cuda`; explicitly use `cpu` or `mps` for development | `cuda` |
| `MODEL_CACHE_DIR` | `./.model-cache` | `/runpod-volume/models/laya` |
| `HF_HOME` | Hugging Face default | `/runpod-volume/huggingface` |
| `HF_TOKEN` | Optional; model is public | Optional runtime secret |

Keep the image's persistent cache paths for Runpod. The mounted `/runpod-volume` must already exist and be writable; the worker refuses an ephemeral fallback. The pinned model revision is `7c76b622dfc5cac71b2dc1c29873efe2ce509a05`. Initialization retains the weights, local encoder configuration, and tokenizer together; a shared file lock protects downloads and tokenizer compatibility updates. An atomic completion marker allows subsequent workers to load without contacting Hugging Face. A failed initialization never publishes a new ready marker. Do not delete or modify cache files while workers use them.

### Deploy from GitHub

Use Runpod's [GitHub integration](https://docs.runpod.io/serverless/workers/github-integration) with a **Queue** endpoint:

1. Publish the implementation to `AdamGoodApp/noah`, branch **`queue-worker`**, and track that branch in Runpod. This keeps the old `main`-based endpoint live until the Queue replacement is verified; fast-forward `main` only after retiring the old endpoint. Authorize Runpod's GitHub App for this repository only.
2. Reuse the existing **10GB standard network volume `noah-models` in `US-IL-1`**; do not create a duplicate or change storage tier.
3. Create a Queue endpoint named **`noah-api`** from that GitHub repository using the root `Dockerfile`. Do not reuse a Load Balancer endpoint ID as a Queue ID.
4. Select one **RTX A5000 (24GB)** GPU in `US-IL-1`. The MCP/API uses pool `AMPERE_24`; select only A5000 by excluding other current members of that pool. Re-check pool membership and stock when configuring it.
5. Attach `noah-models`. Keep the image's cache paths and `LAYA_DEVICE=cuda`; configure no HTTP ports and no client bearer key in the worker.
6. Set minimum workers **0**, maximum workers **1**, idle timeout **5 seconds**, **`QUEUE_DELAY` scaling with a target of 1 second**, FlashBoot enabled, and container disk **10GB**. Keep handler concurrency at one.
7. Wait for the GitHub image build, then submit the public synthetic job above. Inspect job status and worker logs to verify successful CUDA initialization and inference without CPU fallback. Let the endpoint scale to zero and verify a subsequent worker reports a cache hit. Inspect the Builds tab to confirm the deployed commit rather than assuming source publication means deployment succeeded.

**Storage and location:** a [network volume](https://docs.runpod.io/storage/network-volumes) restricts workers to its data center. This configuration uses standard storage and a 24GB GPU in `US-IL-1`. On 2026-09-18, the catalog quoted A5000 Serverless at **$0.69 per running worker-hour**, and the console quoted the existing 10GB standard volume at **$0.70/month** ($0.07/GB). Storage remains billable at zero workers. Pricing and stock can change.

Runpod's [beta Global Volumes](https://docs.runpod.io/storage/globalvolume) are currently documented for Pods, and the connected Serverless tools expose no Global Volume attachment. They also lack file locking and atomic rename. **Do not point this cache initializer at a Global Volume.** Keep a POSIX-capable regional network volume for this Serverless worker.

### Worker regression checks

```bash
python -m pip install pytest
python -m pytest tests/test_handler.py tests/test_model_cache.py
```

These isolated checks cover job validation, complete results, sanitized failures, serialized inference, startup ordering, and cache failures. They do not replace a real-model SDK smoke test or verification on an actual CUDA worker.

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
