# RadioConsistency

Consistency-model experiments for **radio-map generation & reconstruction** on Compute Canada (Narval).  
This codebase is **derived from** OpenAI’s `consistency_models` but **significantly modified** for radio-map data, training/eval flows, and SLURM usage.

> Attribution: Portions of this repository are adapted from
> [OpenAI/consistency_models](https://github.com/openai/consistency_models) (MIT).

---

## ✨ What’s different vs. the original
- **Domain shift**: radio-map images (custom loaders & preprocessing).
- **Models**: UNet variants for radio data (`cm/unet_radio.py`, `cm/unet_conditional.py`).
- **Training**: task-specific flags & entry points (`cm/test.py`, `scripts/cm_train_radio.py`).
- **Cluster-friendly**: SLURM/Narval scripts for training, resume, conditional runs, and sampling.
- **Eval path**: NPZ sample dumps + utilities under `evaluations/`.

---

## 🚀 Quickstart

### On Compute Canada (Narval)
```bash
module load git-lfs/3.4.0
python -m venv .venv && source .venv/bin/activate
pip install -e .
````

### Local (Linux/macOS)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## 📦 Datasets

Keep large data **out of git**. See `datasets/README.md` for notes.
Example preprocessing:

```bash
python prepare_data.py --in /path/to/raw --out datasets/processed
```

---

## 🏋️ Training

```bash
bash ct_radiomap.sh
# or on SLURM
sbatch scripts/train_radiomap.sbatch
```

Resume:

```bash
bash ctradiomap_resume.sh checkpoints/last.pt
```

Conditional:

```bash
bash ctradiomapcon.sh
```

---

## 📸 Sampling

```bash
bash ctsampling.sh checkpoints/last.pt
```

---

## 📈 Evaluation

```bash
python -m evaluations.evaluator --in samples/out.npz --metrics fid,precision,recall
```

---

## 📁 Repository Layout

```
cm/          # models, losses, trainers
scripts/     # sbatch helpers
datasets/    # dataset notes only
evaluations/ # metrics
checkpoints/ # weights (ignored / LFS)
docker/      # optional
```

---

## 📜 License & Attribution

* MIT (see LICENSE).
* This project includes modified code from OpenAI’s `consistency_models` (MIT). See `NOTICE`.

---

## 📝 Citation

```
@software{RadioConsistency,
  author = {Paria Mohammadzadeh},
  title  = {RadioConsistency: Consistency-Model Experiments for Radio Maps},
  year   = {2025},
  url    = {https://github.com/pariamdz1992/RadioConsistency}
}
```

