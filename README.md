# NeuroPlan

NeuroPlan turns questions about mouse visual cortex into typed analysis plans, runs them on Allen Visual Coding Neuropixels data, saveing the evidence needed to replay each result. This is done using `pydantic` along with using `vLLM` to serve a local LLM. We run through an example using `Qwen2.5-3B` model, although the project support other LLMs, and cloud models. 

## Inference


We will use [uv](https://docs.astral.sh/uv/) and use Python 3.12. Install `vllm` and `vllm-gguf-plugin` for serving local `gguf` files with: 
```bash
uv pip install vllm
uv pip install vllm-gguf-plugin
```
Now initialize the data, and run through the tests. 
```bash
neuroplan_init(){
  uv sync --locked
  uv run --locked neuroplan prepare
  uv run --locked neuroplan run-plan examples/visp_visl.json --out runs/paired
  uv run --locked neuroplan replay runs/paired/run.json --out runs/replay
  uv run --locked neuroplan benchmark --backend keyword --out runs/keyword
  uv run --locked python -m unittest discover -s tests -v
}
neuroplan_init
```

The first data preparation fetches about 106 MiB from the Allen public bucket. It caches the five source files under `data/raw/` and checks each file against its pinned SHA-256. If you already downloaded the original CSVs, point `prepare --raw-dir` to their directory. Analysis and replay use the prepared CSV and manifest and can run offline. Output commands expect a fresh run directory.

The included [paired example](reports/paired/figure.png) uses 24 recording sessions, with 1,502 VISp units and 933 VISl units contributing. The mean across sessions of the within-session median gOSI differences (VISp minus VISl) is **+0.01241** on the gOSI scale. This describes the units and sessions retained by this plan. It does not estimate the uncertainty of a population effect.

Get the model, for example, `Qwen2.5-3B`, and serve 
```bash
curl -fL -C - -o qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

vllm_qwen_gguf_serve(){
  vllm serve \
    ./qwen2.5-3b-instruct-q4_k_m.gguf \
    --tokenizer Qwen/Qwen2.5-3B-Instruct \
    --served-model-name qwen-gguf \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 2048 --max-num-seqs 1 --enforce-eager
}

vllm_qwen_gguf_serve
```
The query `env -u VIRTUAL_ENV VLLM_BASE_URL=http://127.0.0.1:8000/v1 uv run --locked neuroplan ask --backend vllm --model qwen-gguf "Compare orientation selectivity in V1 and LM within each recording session."   --out runs/vllm-paired` returns
```
{
  "n_sessions": 24,
  "n_units_by_region": {
    "VISl": 3482,
    "VISp": 4307
  },
  "mean_paired_difference": -0.004652200214993276
}
```
with generated results in `runs/vllm-paired`. 


## Understanding the analysis 

To do. 



