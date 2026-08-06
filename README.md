# SkepticMonkey

SkepticMonkey is a standalone API that **generates** LLM responses and returns **line-level uncertainty** scores for every line (code and prose). It uses [LM-Polygraph](https://github.com/IINemo/lm-polygraph) with an [LLM Uncertainty Head](https://github.com/datapaf/llm-uncertainty-head).

## Install

```bash
git clone https://github.com/datapaf/SkepticMonkey.git
cd SkepticMonkey

conda create -n skeptic-monkey python=3.11 -y
conda activate skeptic-monkey

pip install -r requirements.txt
```

Download the pretrained uncertainty head (private Hugging Face repo — run `hf auth login` first):

```bash
hf download datapaf/instruct_line_diff_tp_hs --local-dir heads/instruct_line_diff_tp_hs
```

Expected layout:

```text
SkepticMonkey/
├── skeptic_monkey/
│   ├── api.py
│   └── service.py
├── heads/instruct_line_diff_tp_hs/
├── requirements.txt
└── README.md
```

## Run

```bash
conda activate skeptic-monkey
uvicorn skeptic_monkey.api:app --host 127.0.0.1 --port 8000
```

Startup loads the base LLM and can take a while. When ready:

- Docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

### Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `SKEPTIC_MONKEY_LLM_ID` | `deepseek-ai/deepseek-coder-1.3b-instruct` | Hugging Face model id |
| `SKEPTIC_MONKEY_HEAD_PATH` | `heads/instruct_line_diff_tp_hs` | Local path to UE head |
| `SKEPTIC_MONKEY_HOST` | `0.0.0.0` | Bind host (when using `python -m`) |
| `SKEPTIC_MONKEY_PORT` | `8000` | Bind port |

## Usage

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Estimate line uncertainties

```bash
curl -X POST http://127.0.0.1:8000/estimate/line \
  -H "Content-Type: application/json" \
  -d "{\"input_text\": \"### Instruction:\\nFix the bug.\\n### Response:\\n\"}"
```

Example response:

```json
{
  "input_text": "...",
  "generation_text": "...",
  "lines": [
    {"text": "Here is the fix:\\n", "uncertainty": 0.12},
    {"text": "```java\\n", "uncertainty": 0.08},
    {"text": "  return x;\\n", "uncertainty": 0.77}
  ],
  "generation_tokens": [1, 2, 3],
  "model_path": "deepseek-ai/deepseek-coder-1.3b-instruct",
  "estimator": "LuqClaimEstimatorDummy_claim"
}
```

`lines` always includes **all** lines (prose and code). Clients can choose which lines to highlight.

### Python client

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/estimate/line",
    json={"input_text": "### Instruction:\\nWrite a hello world in Python.\\n### Response:\\n"},
    timeout=600,
)
r.raise_for_status()
data = r.json()
for line in data["lines"]:
    print(f"{line['uncertainty']:.3f}  {line['text']!r}")
```

## License

MIT (unless otherwise noted by bundled dependencies).
