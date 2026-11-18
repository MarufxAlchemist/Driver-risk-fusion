# 02. GPU access (H3)

Local CPU runs are enough for P0 and for feature-based models. Clip models (3D-CNN, 2D-CNN+LSTM, P2-A2/A3) need a GPU.

## Options, in order of effort

1. **Local laptop GPU.** This machine has an NVIDIA GeForce RTX 3050 Laptop GPU. Check its memory: `nvidia-smi` (a 4 GB card is enough for small clips such as 8 frames at 96x96, not for large batches). The repo installs the **CPU** PyTorch build (ADR-0006); a CUDA build needs a separate environment.
2. **Google Colab** (free tier GPU, sessions end after some hours): use [colab_train.ipynb](colab_train.ipynb).
3. **Kaggle notebooks** (weekly GPU quota): the same notebook works; enable the GPU in the notebook settings.
4. **College lab:** ask the guide in the week-2 meeting.

## Using the notebook

- [ ] Upload `docs/human_tasks/colab_train.ipynb` to Colab (File > Upload notebook) and choose Runtime > Change runtime type > GPU.
- [ ] The repository is cloned from GitHub. If it is private, create a token with read access and paste it where the notebook says; never save the token in the notebook.
- [ ] Run the cells top to bottom. The first run is a **smoke test** (`scripts/demo_stub.py` on stub data).
- [ ] Data for real runs: put processed parquet (features, not video) on Google Drive and mount it; raw video of people must not be uploaded to a shared notebook.
- [ ] `scripts/train.py` does not exist yet (it arrives with P2-C3). The notebook has a cell for it that is marked as waiting.

## Honest status

The notebook has **not been run on Colab or Kaggle** (no access from the development environment). The first run may need small fixes, for example the CUDA PyTorch index URL in cell 4. Tell Claude what failed.

## Record per run

GPU type and hours per run go in the run log (`04_experiment_protocol.md` section 6).
