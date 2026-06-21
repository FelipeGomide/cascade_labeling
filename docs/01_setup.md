# Setup

## Hardware assumptions

Developed and tuned for:
- NVIDIA RTX 4060 Ti, **8 GB VRAM**
- 12 CPU cores, 31 GB RAM

Batch sizes in `configs/models/*.yaml` are chosen to fit comfortably in 8 GB
VRAM. If you have less VRAM, reduce `batch_size` in `bi_encoder.yaml` /
`cross_encoder.yaml` / `rag_label_generator.yaml`. See
[`06_rag_labels.md`](06_rag_labels.md) for a concrete note on batch-size limits
observed for the RAG-label generator.

## Python environment

Two ways to set this up are provided; in practice, development on this machine
used the plain venv route (faster to iterate than solving a full conda env).

### Option A — venv (used during development)

```bash
cd cascade_labeling
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# CPU/data packages
.venv/bin/pip install numpy scipy pandas pyarrow pyyaml requests omegaconf bm25s

# GPU stack (CUDA 12.1 wheel; works with newer drivers too, e.g. driver reporting CUDA 13)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install sentence-transformers faiss-cpu accelerate transformers

# profiling / plotting / notebooks
.venv/bin/pip install psutil nvidia-ml-py codecarbon
.venv/bin/pip install matplotlib seaborn plotly kaleido jupyterlab ipykernel nbformat nbclient

# install this package in editable mode so `cascade.*` imports work everywhere
.venv/bin/pip install -e .
```

Verify GPU visibility:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Option B — conda (`environment.yml`)

```bash
conda env create -f environment.yml
conda activate cascade
pip install -e .
```

`environment.yml` and `requirements.txt` are kept in sync with what's actually
needed; **`pyxclib` is intentionally NOT used** — it requires a Cython build
step that failed in this environment (`ModuleNotFoundError: No module named
'Cython'` at install time). Instead, the same metrics (P@k, nDCG@k, PSP@k,
PS-nDCG@k) are implemented directly in numpy in
`src/cascade/eval/metrics.py` — see [`03_stages_and_models.md`](03_stages_and_models.md).

## Registering the Jupyter kernel (for running notebooks)

```bash
.venv/bin/python -m ipykernel install --user --name=cascade-venv --display-name="cascade venv"
```

Then in JupyterLab/VS Code, select the **"cascade venv"** kernel for the
notebooks in `notebooks/`.

To execute a notebook headlessly (as done for the figures in `reports/figures/`):

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cascade-venv 01_dataset_eda.ipynb
```
