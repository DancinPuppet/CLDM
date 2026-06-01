# CLDM: Correlation-Aware Latent Diffusion Model for Source Localization on Social Interaction Graphs

This repository contains the official implementation of **CLDM**, a feature-based framework for source localization in social networks. CLDM operates solely on graph structural topology without requiring user profile information, and introduces a correlation-aware latent space design with single-step diffusion-based representation enhancement.

> **Paper:** *Correlation-Aware Latent Diffusion Model for Source Localization on Social Interaction Graphs*
> Submitted to CIKM 2026 (Submission ID: 1932)

---

## Repository Structure

```
.
├── args.py                 # Hyperparameter configuration
├── create_graphs.py        # Dataset loading and graph construction
├── data.py                 # PyG data conversion and dataset class
├── node_feature.py         # Structural feature computation
├── model.py                # CLDM model architecture
├── train.py                # Training loop with early stopping
├── eval.py                 # Evaluation metrics
├── main.py                 # Entry point for training and testing
├── static.py               # Dataset statistics computation
└── data/
    └── saved_graphs/           # Preprocessed graph cache
        ├── twitter25_graph.pkl # Proposed dataset (Twitter25)
        ├── twitter15_graph.pkl # Twitter15 benchmark
        ├── twitter16_graph.pkl # Twitter16 benchmark
        └── weibo_graph.pkl     # Weibo benchmark
```

---

## Requirements

- Python 3.10
- PyTorch 2.5.1 (CUDA 12.1)
- torch-geometric 2.6.1
- networkx 2.8.8
- scikit-learn 1.7.0
- numpy 1.26.4
- ndlib 5.1.1
- tqdm
- scipy 1.15.2

Install core dependencies:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1
pip install networkx scikit-learn numpy ndlib scipy tqdm
```

---

## Configuration

All hyperparameters are managed in `args.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `graph_type` | `weibo` | Dataset selection: `twitter25`, `twitter15`, `twitter16`, `weibo` |
| `epochs` | `300` | Maximum training epochs |
| `batch_size` | `16` | Training batch size |
| `lr` | `1e-4` | Initial learning rate |
| `latent_dim` | `16` | Latent space dimensionality |
| `hidden_channels` | `32` | GAT hidden dimension |
| `heads` | `4` | GAT attention heads |
| `kl_start_weight` | `0.001` | Initial KL annealing weight |
| `kl_end_weight` | `0.1` | Final KL annealing weight |
| `beta_start` | `1e-4` | Diffusion noise schedule start |
| `beta_end` | `0.02` | Diffusion noise schedule end |
| `max_time_steps` | `1000` | Diffusion timesteps |

---

## Training

Set the target dataset in `args.py`:

```python
self.graph_type = 'twitter25'  # or twitter15, twitter16, weibo
self.load_model = False
```

Then run:

```bash
python main.py
```

Training logs are saved to `output/{model}_{dataset}_train_log.txt`. The best model (by validation F1) is saved to `model_saves/checkpoints/{model}_{dataset}_best_f1.pt`.

---

## Evaluation

To evaluate a trained model, set in `args.py`:

```python
self.load_model = True
```

Then run:

```bash
python main.py
```

This loads the best checkpoint and reports Accuracy, F1, Precision, and Recall on the test set.

---

## Dataset Statistics

To reproduce the radar chart statistics (clustering coefficient, Gini degree, max shortest path length) reported in the paper:

```bash
python static.py
```

Requires all four datasets to be preprocessed and cached beforehand.

---

## License

This project is released for research purposes. The Twitter25 dataset is collected in accordance with the Twitter Developer Agreement and is provided for non-commercial academic use only.
