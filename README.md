# DeepFuse: code, splits and results

Code, split files and result files for *Adaptation depth varies across tasks in few-shot transfer from a frozen dermatology foundation model*.

## Contents

| Path | What it holds |
|---|---|
| `code/icd_pauc/` | Feature caching, the DeepFuse selection and fusion head, all twelve comparators, the ablation arms, and the statistics scripts. |
| `splits/` | The split files listed below. They contain no images. |
| `splits/index/` | One index per dataset: image file name, label and split. |
| `scripts/build_splits.py` | Rebuilds the other files in `splits/` from `splits/index/` and the ISIC Archive metadata. |
| `run.sh`, `config.env.example` | One-command entry point (see Quick start). |
| `results/` | The result files behind the paper's tables and figures. |

## Splits

| File | Content |
|---|---|
| `derm7pt_case_split.csv` | The Derm7pt split used in the paper. It has one case-level stratified 65/15/20 split, shared by the dermoscopic and clinical views, with 541, 124 and 169 cases. The partitions shipped per view agree on only 44.8% of the 834 shared cases, so they were replaced. |
| `dermnet_test_contamination.csv` | Flags for each official DermNet test row: filename in train, filename in val, and contradictory label across rows. `masked` marks the 449 rows excluded in the de-duplicated analysis. |
| `bcn20000_test_lesion_overlap.csv` | For each official BCN20000 test image, whether its ISIC `lesion_id` also occurs in train. 90.4% do. |
| `hiba_test_patient_overlap.csv` | For each official HIBA test image, whether its ISIC `patient_id` also occurs in train. 77.2% do. |
| `bcn20000_lesion_disjoint_split.csv`, `hiba_patient_disjoint_split.csv` | Lesion-disjoint (BCN20000) and patient-disjoint (HIBA) resplits at the official proportions, built with `GroupShuffleSplit` and seed 0. They are provided for future work. The paper does not evaluate on them: it keeps the official splits and excludes these two datasets from every primary aggregate. |

The splits list file names and labels only.

## Quick start

Everything runs from a shell with Python 3.10.

```bash
pip install -r requirements.txt      # PyTorch 2.4.1 + CUDA 11.8, timm 0.9.16, scikit-learn 1.7.2, ...
./run.sh tables                      # recompute every table from results/ (no GPU, no images, ~1 min)
```

To re-run the experiments, set the paths once and use the other modes:

```bash
cp config.env.example config.env     # then edit the five paths in it
./run.sh check                       # DDI only, K=4, 2 seeds: a few minutes on one GPU
./run.sh all                         # every experiment, then the statistics (one GPU, long)
./run.sh splits                      # optional: rebuild splits/ from splits/index/ and ISIC metadata
```

| Mode | Needs | What it does |
|---|---|---|
| `tables` | Python only | Runs the five statistics scripts on the shipped `results/` and rewrites the summary files there. |
| `check` | GPU, PanDerm, data for DDI | Builds the DDI caches and runs every experiment script on DDI with 2 seeds. Each per-seed number is compared with the shipped results, which must match exactly; it prints `PASS` or `FAIL` and then deletes its own outputs. |
| `all` | GPU, PanDerm, all ten datasets, LibFewShot, about 60 GB of disk | Moves the shipped results to `results_shipped/`, builds all feature caches, runs every experiment one after another, then the statistics. |
| `splits` | ISIC Archive `metadata.csv` | Rebuilds the six split files in `splits/` from `splits/index/`. |

Paths in `config.env`:

| Variable | Value |
|---|---|
| `PANDERM_CODE` | A clone of <https://github.com/SiyuanYan1/PanDerm>. Its `datasets/derm_data.py` and `models/modeling_finetune.py` are used unmodified and are not redistributed here (CC BY-NC-ND 4.0). |
| `PANDERM_CKPT` | The PanDerm ViT-L/16 checkpoint `panderm_ll_data6_checkpoint-499.pth`, from the PanDerm repository. DINOv2 ViT-L/14 and CLIP ViT-L/14 are downloaded by `timm` on first use. |
| `DEEPFUSE_DATA` | The folder holding the dataset images (see below). |
| `LIBFEWSHOT` | A clone of <https://github.com/RL-VIG/LibFewShot>, for the few-shot comparators. |
| `ISIC_META` | The ISIC Archive `metadata.csv`, only for `./run.sh splits`. |

## Dataset layout

`splits/index/<dataset>.csv` lists every image with its label and its train, val or test split, in the order the code reads them. The files contain no images.

The images are those of the **pre-processed public datasets released by the PanDerm authors** (download links in the "Using Pre-processed Public Datasets" section of the [PanDerm README](https://github.com/SiyuanYan1/PanDerm)). Their folder structure and file names can differ from the original releases (Derm7pt, for example, is a single flat folder), so use these downloads rather than the original websites, and follow each dataset's licence (BCN20000: CC BY-NC; HIBA: CC BY; Derm7pt and DDI: registration with the providers). Every file name in our indexes occurs in the CSV shipped with the corresponding PanDerm download. Our indexes keep PanDerm's rows, labels and splits, except for Derm7pt, which uses one case-level split shared by both views (`splits/derm7pt_case_split.csv`), and HIBA, MSKCC and DDI, whose labels are mapped from the diagnosis columns as described in `code/icd_pauc/model_zoo.py` (two HIBA lichenoid-keratosis images are dropped).

Unpack each download under `$DEEPFUSE_DATA` so that each file in the `image` column exists in the folder below:

| Dataset key | Index | Image folder under `$DEEPFUSE_DATA` | Example file |
|---|---|---|---|
| `ham` | `ham.csv` | `HAM10000_clean/ISIC2018/` | `ISIC_0027419.jpg` |
| `pad` | `pad.csv` | `pad-ufes/images/` | `PAT_999_20_401.png` |
| `dermnet` | `dermnet.csv` | `Dermnet/images/` | `polymorphous-light-eruption-10.jpg` |
| `bcn` | `bcn.csv` | `BCN20000/images/` | `ISIC_0053454.JPG` |
| `hiba` | `hiba.csv` | `HIBA/images/` | `ISIC_0079358.JPG` |
| `mskcc` | `mskcc.csv` | `MSKCC/images/` | `ISIC_0009868.JPG` |
| `ddi` | `ddi.csv` | `DDI/images/` | `000001.png` |
| `patch16` | `patch16.csv` | `PATCH16/images/` | `nontumor_skin_chondraltissue_chondraltissue_ws_318_48.jpg` |
| `derm7pt_derm`, `derm7pt_clin` | `derm7pt_derm.csv`, `derm7pt_clin.csv` | `Derm7pt/images/` | `nal099.jpg` |

## Running the steps by hand

`run.sh` is a plain shell script; each line is one command and can be run on its own, from a terminal or an IDE such as PyCharm (interpreter: the environment with `requirements.txt`). No environment variables are needed: importing `icd_pauc` reads `config.env` at the repository root and puts `$PANDERM_CODE/classification` on the Python path. For example, the main table (Table 2) is `python -m icd_pauc.revision_main --all --norm transductive --tag trans` from `code/`, followed by `python icd_pauc/revision_tables.py --tag trans ind`. Feature caches go to `code/icd_pauc/cache/` (`zoo_*`: the default read-out, the class token after the final LayerNorm; `deep_*`: the 96 streams, 24 blocks × 4 poolings). Results go to `results/`. The runs for the Table 3(b) symmetric row, the DermNet de-duplication check and the budget comparison predate the revision and rank streams with seeds 0-4. `revision_clinical.py` reads the per-image predictions written by `revision_main.py` and `fsl_heads_run.py --save_preds`; these are not shipped, so it runs only after the experiments.

Results should match the shipped files. A full re-run may differ in the last digits, because GPU arithmetic is not always deterministic; the `check` mode reproduced every per-seed number exactly on our machine (one NVIDIA RTX 4090).

## Where each table comes from

| Paper item | Result files | Script |
|---|---|---|
| Table 2, Table 4 and the envelope tests | `revision_main_{trans,ind}.json`, `fsl_heads_rev_{trans,ind}*.json`, which feed `revision_tables_{trans,ind}.json` | `revision_tables.py` |
| Table 3(a) and Fig. 2 | `revision_depth_report_{panderm,dinov2,clip}_{trans,ind}.json` | `revision_depth_report.py` |
| Table 3(b) | `revision_main_{panderm,dinov2,clip}_abl_{trans,ind}*.json`, which feed `revision_stats_backbones.json`; the symmetric row uses `picks_symmetric_sym10.json` and `revision_stats_clusters.json` | `revision_stats_backbones.py`, `revision_stats_clusters.py` |
| Clinical metrics and McNemar tests | `revision_clinical_trans.json` | `revision_clinical.py` |
| Class count versus best block (Section 4.1) | `revision_depth_report_{panderm,dinov2,clip}_{trans,ind}.json` | `class_count_depth.py` |

## License

The code in this repository is released under the MIT License (`LICENSE`). The split and index files list file names and labels of third-party datasets; the images themselves are not included and remain under each dataset's own licence. The PanDerm code and checkpoint are not included and are used under their own terms (CC BY-NC-ND 4.0).
