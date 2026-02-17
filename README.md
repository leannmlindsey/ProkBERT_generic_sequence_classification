# The ProkBERT model family

ProkBERT is an advanced genomic language model specifically designed for microbiome analysis. This repository contains the ProkBERT package and utilities, as well as the LCA tokenizer and model definitions.


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Installing with pip](#installing-with-pip)
  - [Installing with conda](#installing-with-conda)
  - [Using Docker](#using-docker)
  - [Using Singularity (Apptainer)](#using-singularity-apptainer)
- [Applications](#applications)
- [Quick start](#quick-start)
- [Tutorials and examples](#tutorials-and-examples)
  - [Tokenization and segmentation](#tokenization-and-segmentation)
  - [Visualizing sequence representations](#visualizing-sequence-representations)
  - [Finetuning example for promoter sequences](#finetuning-example-for-promoter-sequences)
  - [Evaluation and inference example](#Evaluation-and-inference-example)
  - [Pretraining example](#pretraining-example)
- [ProkBERT's results](#prokbert's-results)
- [Citing this work](#citing-this-work)


### Introduction
The ProkBERT model family is a transformer-based, encoder-only architecture based on [BERT](https://github.com/google-research/bert). Built on transfer learning and self-supervised methodologies, ProkBERT models capitalize on the abundant available data, demonstrating adaptability across diverse scenarios. The models’ learned representations align with established biological understanding, shedding light on phylogenetic relationships. With the novel Local Context-Aware (LCA) tokenization, the ProkBERT family overcomes the context size limitations of traditional transformer models without sacrificing performance or the information-rich local context. In bioinformatics tasks like promoter prediction and phage identification, ProkBERT models excel. For promoter predictions, the best-performing model achieved an MCC of 0.74 for E. coli and 0.62 in mixed-species contexts. In phage identification, they all consistently outperformed tools like VirSorter2 and DeepVirFinder, registering an MCC of 0.85. Compact yet powerful, the ProkBERT models are efficient, generalizable, and swift.

### Features
- Tailored to microbes. 
- Local Context-Aware (LCA) tokenization for better genomic sequence understanding.
- Pre-trained models available for immediate use and fine-tuning.
- High performance in various bioinformatics tasks.
- Facilitation of both supervised and unsupervised learning.


## Installation

### Installing with pip

The recommended way to install ProkBERT is through pip, which will handle most dependencies automatically:

```bash
pip install git+https://github.com/nbrg-ppcu/prokbert.git
```

### Installing with conda
(The best is using the github codes)
ProkBERT is also available as a conda package from the Bioconda channel. As first step it is reccomended to install the CUDA enabled pytorch:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```


To install it using conda, run:

```bash
conda install prokbert -c bioconda
```

### Using Docker
Before using the ProkBERT container with GPU support, make sure you have the following installed on your system:
- [Docker](https://docs.docker.com/get-docker/) (required if you plan to use the Docker image)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (required if you intend to use Docker with GPU support)

To pull and run the ProkBERT Docker image, use:

```bash
docker pull obalasz/prokbert
```

To run the container with GPU support, use:

```bash
docker run --gpus all -it --rm -v $(pwd):/app obalasz/prokbert bash
```

### Using Singularity (Apptainer)
To pull directly from Docker Hub and convert to a Singularity image file:
```bash
singularity pull prokbert.sif docker://obalasz/prokbert
```

Once you have your `.sif` file, you can run ProkBERT with the following command:
```bash
singularity run --nv prokbert.sif bash
```



## Applications
ProkBERT has been validated in several key genomic tasks, including:
- Learning meaningful representation for seqeuences (zero-shot capibility)
- Accurate bacterial promoter prediction.
- Detailed phage sequence analysis within complex microbiome datasets.


## Quick Start
Our models and datasets are available on the [hugginface page](https://huggingface.co/neuralbioinfo). 
The models are easy to use with the [transformers](https://github.com/huggingface/transformers) package.
We provide examples and descriptions as notebooks in the next chapter and some example scsripts regarging how to preprocess your sequence data and how to finetune the available models. The examples are available in the [example](https://github.com/nbrg-ppcu/prokbert/tree/main/examples) folder of this repository. 

### TLDR example
To load the model from Hugging Face:
```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
model = AutoModel.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)

segment = "TATGTAACATAATGCGACCAATAATCGTAATGAATATGAGAAGTGTGATATTATAACATTTCATGACTACTGCAAGACTAA"

# Tokenize the input and return as PyTorch tensors
inputs = tokenizer(segment, return_tensors="pt")

# Pass the tokenized input to the model
outputs = model(**inputs)
```

## Tutorials and examples:


### Visualizing sequence representations
An example of how to visualize the genomic features of ESKAPE pathogens. More description about the dataset is available on Hugging Face
Example:
 - ESKAPE pathogen genomic features: [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Embedding_visualization.ipynb) 

### Finetuning example for promoter sequences
Here we provide an example of a practical transfer learning task. It is formulated as a binary classification. We provide a notebook for presenting the basic concepts and a command line script as a template. 
Examples:
- Finetuning for promoter identification task: [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Finetuning.ipynb)
- Python script for the finetuning: [link](https://github.com/nbrg-ppcu/prokbert/blob/main/examples/finetuning.py)

### Tokenization and segmentation
For examples of how to preprocess the raw sequence data, which are frequently stored in fasta format:
Examples:
- Segmentation (sequence preprocessing): [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Segmentation.ipynb)
- Tokenization [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Tokenization.ipynb)
- Preprocessing for pretraining
  
  
Usage example:
```bash
git clone https://github.com/nbrg-ppcu/prokbert
cd examples
python finetuning.py \
  --model_name neuralbioinfo/prokbert-mini \
  --ftmodel mini_promoter \
  --model_outputpath finetuning_outputs \
  --num_train_epochs 1 \
  --per_device_train_batch_size 128 
```
For practical applications or for larger training tasks we recommend using the [Distributed DataParallel](https://huggingface.co/docs/transformers/en/perf_train_gpu_many). 

### Evaluation and inference example

In this section, we provide a practical example for evaluating finetuned ProkBERT models. This is crucial for understanding the model's performance on specific tasks, such as promoter prediction or phage identification. 

We have prepared a detailed example that guides you through the process of evaluating our finetuned models. This includes loading the model, preparing the data, running the inference, and interpreting the results.

Example:
- [Evaluation of the Finetuned Models](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Inference.ipynb)


### Pretraining example

Here you can find an example of pretraining ProkBERT from scratch. Pretraining is an essential step, allowing the model to learn the underlying patterns before being fine-tuned for downstream tasks. All of the pretrained models are available on Hugging Face.

#### Pretrained Models:

| Model | k-mer | Shift | Hugging Face URL |
| ----- | ----- | ----- | ---------------- |
| ProkBERT-mini | 6 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini) |
| ProkBERT-mini-c | 1 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-c) |
| ProkBERT-mini-long | 6 | 2 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-long) |

#### Preprocessing the example data:

It is a good practice to preprocess larger sequence sets (>100MB) in advance. Below is an example script for achieving this by running on multiple fasta files:

Clone the ProkBERT repository and navigate to the examples directory. Then, preprocess the example data into a format suitable for training. This involves converting sequences into k-mer representations. Run the following commands:

```bash
git clone https://github.com/nbrg-ppcu/prokbert
cd prokbert/examples
python prokbert_seqpreprocess.py \
  --kmer 6 \
  --shift 1 \
  --fasta_file_dir ../src/prokbert/data/pretraining \
  --out ../src/prokbert/data/preprocessed/pretraining_k6s1.h5
```

Parameters:
- `--kmer`: The size of the k-mer (number of bases) to use for sequence encoding.
- `--shift`: The shift size for sliding the k-mer window across sequences.
- `--fasta_file_dir`: Directory containing your FASTA files for pretraining.
- `--out`: Output file path for the preprocessed data in HDF5 format.

#### Running the pretraining from scratch:

Use the preprocessed HDF file as input for pretraining. Execute the commands below:

```bash
python prokbert_pretrain.py \
  --kmer 6 \
  --shift 1 \
  --dataset_path ../src/prokbert/data/preprocessed/pretraining_k6s1.h5 \
  --model_name prokbert_k6s1 \
  --output_dir ./tmppretraining \
  --model_outputpath ./tmppretraining
```

Parameters:
- `--model_name`: Name for the model configuration to be used or saved.
- `--output_dir`: Directory where the training logs and temporary files will be saved.
- `--model_outputpath`: Path where the final trained model should be saved.

## Running the Scripts

### Available Models

| Model | HuggingFace Name | k-mer | Shift | Max Position Embeddings |
| ----- | ---------------- | ----- | ----- | ----------------------- |
| ProkBERT-mini | `neuralbioinfo/prokbert-mini` | 6 | 1 | 1024 |
| ProkBERT-mini-c | `neuralbioinfo/prokbert-mini-c` | 1 | 1 | 2048 |
| ProkBERT-mini-long | `neuralbioinfo/prokbert-mini-long` | 6 | 2 | 2048 |

### 1. Embedding Analysis

Extracts embeddings from a pre-trained ProkBERT model and evaluates them using:
- Linear probe (logistic regression)
- 3-layer neural network
- Silhouette score (embedding quality)
- PCA visualization

**Input:** A directory containing `train.csv`, `dev.csv` (or `val.csv`), and `test.csv`. Each CSV must have `sequence` and `label` columns.

#### Running directly with Python

```bash
python embedding_analysis_prokbert.py \
  --csv_dir /path/to/csv_data \
  --model_path neuralbioinfo/prokbert-mini \
  --max_length 1024
```

#### Running all 3 models

```bash
# prokbert-mini (kmer=6, shift=1, max 1024 tokens)
python embedding_analysis_prokbert.py \
  --csv_dir /path/to/csv_data \
  --model_path neuralbioinfo/prokbert-mini \
  --max_length 1024

# prokbert-mini-c (kmer=1, shift=1, max 2048 tokens)
python embedding_analysis_prokbert.py \
  --csv_dir /path/to/csv_data \
  --model_path neuralbioinfo/prokbert-mini-c \
  --max_length 2048

# prokbert-mini-long (kmer=6, shift=2, max 2048 tokens)
python embedding_analysis_prokbert.py \
  --csv_dir /path/to/csv_data \
  --model_path neuralbioinfo/prokbert-mini-long \
  --max_length 2048
```

Results are saved to `./results/embedding_analysis/<model_name>/` including metrics JSON, embeddings `.npz`, PCA plots, and trained classifiers. If embeddings have already been extracted (the `.npz` file exists), they are loaded from cache and extraction is skipped.

#### Running on HPC with SLURM

Three scripts are provided in `slurm_scripts/`:

1. **`wrapper_run_embedding_analysis.sh`** — Edit the configuration section (`CSV_DIR`, `MODEL_PATH`, etc.) then run:
   ```bash
   bash slurm_scripts/wrapper_run_embedding_analysis.sh
   ```
   This submits an SBATCH job requesting 1 A100 GPU, 64 GB memory, 4 hours.

2. **`run_embedding_analysis_interactive.sh`** — For interactive GPU sessions (e.g. `sinteractive --gres=gpu:1`), runs directly on the current node:
   ```bash
   bash slurm_scripts/run_embedding_analysis_interactive.sh
   ```
   Sources configuration from the wrapper script automatically.

#### All options

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--csv_dir` | (required) | Directory containing train/dev/test CSV files |
| `--model_path` | `neuralbioinfo/prokbert-mini` | HuggingFace model name or local path |
| `--output_dir` | `./results/embedding_analysis` | Base output directory (model name appended automatically) |
| `--batch_size` | 32 | Batch size for embedding extraction |
| `--max_length` | 1024 | Max sequence length in base pairs (clamped to model max if exceeded) |
| `--pooling` | `mean` | Pooling strategy: `mean`, `max`, or `cls` |
| `--nn_epochs` | 100 | Training epochs for the 3-layer NN |
| `--nn_hidden_dim` | auto | Hidden dim for the 3-layer NN (defaults to model embedding dim) |
| `--nn_lr` | 0.001 | Learning rate for the 3-layer NN |
| `--seed` | 42 | Random seed |
| `--include_random_baseline` | off | Also evaluate a randomly initialized model as baseline |

### 2. Fine-tuning

Fine-tunes a ProkBERT model for binary classification on a CSV dataset. Supports early stopping, mixed-precision training, and multiple replicates with different seeds.

**Input:** A directory containing `train.csv`, `dev.csv` (or `val.csv`), and `test.csv`. Each CSV must have `sequence` and `label` columns.

#### Running directly with Python

```bash
python finetune_prokbert_phage.py \
  --dataset_dir /path/to/csv_data \
  --model_name neuralbioinfo/prokbert-mini \
  --max_length 1024 \
  --output_dir ./results/csv_binary/my_dataset \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 32 \
  --early_stopping_patience 3 \
  --fp16
```

#### Running all 3 models

```bash
# prokbert-mini (kmer=6, shift=1, max 1024 tokens)
python finetune_prokbert_phage.py \
  --dataset_dir /path/to/csv_data \
  --model_name neuralbioinfo/prokbert-mini \
  --max_length 1024 \
  --output_dir ./results/csv_binary/my_dataset/prokbert-mini \
  --fp16

# prokbert-mini-c (kmer=1, shift=1, max 2048 tokens)
python finetune_prokbert_phage.py \
  --dataset_dir /path/to/csv_data \
  --model_name neuralbioinfo/prokbert-mini-c \
  --max_length 2048 \
  --output_dir ./results/csv_binary/my_dataset/prokbert-mini-c \
  --fp16

# prokbert-mini-long (kmer=6, shift=2, max 2048 tokens)
python finetune_prokbert_phage.py \
  --dataset_dir /path/to/csv_data \
  --model_name neuralbioinfo/prokbert-mini-long \
  --max_length 2048 \
  --output_dir ./results/csv_binary/my_dataset/prokbert-mini-long \
  --fp16
```

Results are saved to the specified `--output_dir`, organized as `./results/csv_binary/<dataset_name>/lr-<lr>_batch-<batch>/seed-<seed>/` when using the SLURM scripts.

#### Running on HPC with SLURM

Three scripts are provided in `slurm_scripts/`:

1. **`wrapper_run_prokbert_csv.sh`** — Edit the configuration section (`CSV_DIR`, `MODEL_NAME`, `DATASET_NAME`, etc.) then run:
   ```bash
   bash slurm_scripts/wrapper_run_prokbert_csv.sh
   ```
   This submits an SBATCH job requesting 1 A100 GPU, 32 GB memory, 24 hours. Set `NUM_REPLICATES` to run multiple seeds (1, 2, 3, ...) as separate jobs.

2. **`run_prokbert_csv_interactive.sh`** — For interactive GPU sessions (e.g. `sinteractive --gres=gpu:1`), runs directly on the current node:
   ```bash
   bash slurm_scripts/run_prokbert_csv_interactive.sh
   ```
   Sources configuration from the wrapper script automatically. Runs all replicates sequentially.

#### All options

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--dataset_dir` | None | Directory containing train/dev/test CSV files (overrides `--dataset_name`) |
| `--model_name` | `neuralbioinfo/prokbert-mini` | HuggingFace model name or local path |
| `--max_length` | 1024 | Max sequence length in base pairs (1024 for mini, 2048 for mini-c/mini-long) |
| `--output_dir` | `./prokbert_phage_finetuned` | Output directory for model checkpoints and results |
| `--num_train_epochs` | 3 | Number of training epochs |
| `--per_device_train_batch_size` | 32 | Batch size per GPU for training |
| `--per_device_eval_batch_size` | 64 | Batch size per GPU for evaluation |
| `--learning_rate` | 1e-4 | Learning rate |
| `--weight_decay` | 0.01 | Weight decay |
| `--warmup_ratio` | 0.1 | Warmup ratio for learning rate scheduler |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `--seed` | 42 | Random seed |
| `--fp16` | off | Use mixed-precision training |
| `--early_stopping_patience` | 3 | Early stopping patience (epochs without improvement) |
| `--save_total_limit` | 2 | Maximum number of checkpoints to keep |
| `--eval_strategy` | `epoch` | Evaluation strategy: `no`, `steps`, or `epoch` |
| `--save_strategy` | `epoch` | Save strategy: `no`, `steps`, or `epoch` |
| `--logging_steps` | 100 | Log every N update steps |
| `--metric_for_best_model` | `eval_mcc` | Metric for selecting best model |
| `--random_init` | off | Use random initialization instead of pre-trained weights |

### 3. Inference

Two inference scripts are provided, depending on the model source:
- **`inference_lambda.py`** — for local fine-tuned checkpoints (produced by step 2)
- **`inference_hf.py`** — for models hosted on HuggingFace Hub (e.g. `neuralbioinfo/prokbert-mini-c-phage`)

**Input:** A CSV file with a `sequence` column (and optionally `label` for evaluation).

Both scripts support batch inference over multiple CSV files using the SLURM scripts described below.

#### 3a. Inference with a local checkpoint

Use `inference_lambda.py` when you have a fine-tuned checkpoint on disk. You must specify `--base_model` to match the model variant used during fine-tuning.

```bash
python inference_lambda.py \
  --checkpoint_path ./results/csv_binary/my_dataset/prokbert-mini/best_model \
  --base_model neuralbioinfo/prokbert-mini \
  --dataset_file /path/to/test.csv \
  --max_length 1024 \
  --save_metrics
```

**Running on HPC with SLURM** — scripts in `slurm_scripts/`:

1. **`wrapper_run_batch_inference.sh`** — Edit the configuration section (`INPUT_LIST`, `OUTPUT_DIR`, `MODEL_PATH`, `BASE_MODEL`, etc.) then run:
   ```bash
   bash slurm_scripts/wrapper_run_batch_inference.sh
   ```
   `INPUT_LIST` is a text file with one CSV path per line. One SLURM job (1 A100 GPU, 32 GB, 2 hours) is submitted per input file.

2. **`run_batch_inference_interactive.sh`** — For interactive GPU sessions, processes all files sequentially:
   ```bash
   bash slurm_scripts/run_batch_inference_interactive.sh
   ```
   Sources configuration from the wrapper script automatically.

Predictions are saved as `<input_basename>_predictions.csv` in the output directory.

**All options (`inference_lambda.py`):**

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--checkpoint_path` | (required) | Path to the fine-tuned model checkpoint directory |
| `--base_model` | `neuralbioinfo/prokbert-mini` | Base model the checkpoint was fine-tuned from |
| `--dataset` | `leannmlindsey/lambda` | HuggingFace dataset name |
| `--dataset_file` | none | Local CSV/TSV file (overrides `--dataset`) |
| `--split` | `test` | Dataset split to use |
| `--batch_size` | 32 | Batch size for inference |
| `--max_length` | 1024 | Max sequence length |
| `--output_dir` | `inference_results` | Directory to save results |
| `--output_file` | auto | Output filename |
| `--no_labels` | off | Run without labels (prediction only) |
| `--save_metrics` | off | Save metrics to a JSON file |
| `--device` | auto | Force `cuda` or `cpu` |

#### 3b. Inference with a HuggingFace model

Use `inference_hf.py` when using a pre-fine-tuned model hosted on HuggingFace Hub. The model is downloaded automatically.

```bash
python inference_hf.py \
  --model_name neuralbioinfo/prokbert-mini-c-phage \
  --dataset_file /path/to/test.csv \
  --max_length 1024 \
  --save_metrics
```

**Running on HPC with SLURM** — scripts in `slurm_scripts/`:

1. **`wrapper_run_batch_inference_hf.sh`** — Edit the configuration section (`INPUT_LIST`, `OUTPUT_DIR`, `MODEL_NAME`, etc.) then run:
   ```bash
   bash slurm_scripts/wrapper_run_batch_inference_hf.sh
   ```
   Same batch pattern as above: one SLURM job per input CSV file.

2. **`run_batch_inference_interactive_hf.sh`** — For interactive GPU sessions, processes all files sequentially:
   ```bash
   bash slurm_scripts/run_batch_inference_interactive_hf.sh
   ```
   Sources configuration from the wrapper script automatically.

**All options (`inference_hf.py`):**

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--model_name` | `neuralbioinfo/prokbert-mini-c-phage` | HuggingFace model name |
| `--kmer` | auto | K-mer size for tokenizer (auto-detected from model name) |
| `--shift` | auto | Shift size for tokenizer (auto-detected from model name) |
| `--dataset` | none | HuggingFace dataset name or path |
| `--dataset_file` | none | Local CSV/TSV file (overrides `--dataset`) |
| `--split` | `test` | Dataset split to use |
| `--batch_size` | 32 | Batch size for inference |
| `--max_length` | 1024 | Max sequence length |
| `--output_dir` | `inference_results` | Directory to save results |
| `--output_file` | auto | Output filename |
| `--no_labels` | off | Run without labels (prediction only) |
| `--save_metrics` | off | Save metrics to a JSON file |
| `--device` | auto | Force `cuda` or `cpu` |

# ProkBERT's results
ProkBERT is a novel genomic language model family designed for microbiome studies, pretrained on extensive genomic datasets from the NCBI RefSeq database. The pretraining covered a diverse range of organisms, utilizing over 206.65 billion tokens from various genomes, encompassing bacteria, viruses, archaea, and fungi. The final dataset included 976,878 unique contigs from 17,178 assemblies, representing 3,882 distinct genera. Detailed methodology and tokenization strategies are elaborated in our [paper](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1331233/full).

## Genomic features
ProkBERT's embeddings provide insights into genomic structures and relationships, distinguishing between different genomic elements. The models showcase strong generalization abilities, aligning with established biological principles.

![UMAP embeddings of genomic segment representations](assets/Figure5_umaps.jpg)
*Figure 1: UMAP embeddings of genomic segment representations, highlighting the distinction between genomic features.*


## Promoter identification
ProkBERT models have been applied to identify bacterial promoters, demonstrating superior accuracy and robustness compared to existing tools.

![Promoter prediction performance metrics](assets/Figure6_prom_res.png)
*Figure 2: Performance metrics of ProkBERT in promoter prediction.*

ProkBERT models demonstrated high performance with an impressive accuracy and MCC (Matthews Correlation Coefficient) of 0.87 and 0.74, outperforming other established tools such as CNNProm and iPro70-FMWin. This highlights ProkBERT's effectiveness in correctly identifying both promoters and non-promoters, with consistent results across various model variants. The evaluation also included a comparative analysis with newer tools like Promotech and iPromoter-BnCNN.

## Phage prediction
In phage sequence analysis, ProkBERT outperforms traditional methods, proving its efficacy in identifying phage sequences within complex genomic data.

![Comparative analysis of ProkBERT's phage prediction performance](assets/Figure7_phag_res.png)
*Figure 3: Comparative analysis showcasing ProkBERT's performance in phage prediction.*

Our evaluations demonstrate the performance of ProkBERT in classifying phage sequences. It achieves high sensitivity and specificity even in challenging cases where available sequence information is limited. However, this exercise also highlights an inherent limitation of ProkBERT, the restricted context window size. 
In comparative benchmarks with varying short sequence lengths, ProkBERT consistently surpassed established tools like VirSorter2 and DeepVirFinder

## Available models and datasets

### Pretrained models

| Model Name | k-mer | Shift | Hugging Face URL |
| --- | --- | --- | --- |
| `neuralbioinfo/prokbert-mini` | 6 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini) |
| `neuralbioinfo/prokbert-mini-long` | 6 | 2 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-long) |
| `neuralbioinfo/prokbert-mini-c` | 1 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-c) |

### Finetuned models for promoter prediction

| Model Name | k-mer | Shift | Hugging Face URL |
| --- | --- | --- | --- |
| `neuralbioinfo/prokbert-mini-promoter` | 6 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-promoter) |
| `neuralbioinfo/prokbert-mini-long-promoter` | 6 | 2 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-long-promoter) |
| `neuralbioinfo/prokbert-mini-c-promoter` | 1 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-c-promoter) |

### Finetuned models for phage identification

| Model Name | k-mer | Shift | Hugging Face URL |
| --- | --- | --- | --- |
| `neuralbioinfo/prokbert-mini-phage` | 6 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-phage) |
| `neuralbioinfo/prokbert-mini-long-phage` | 6 | 2 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-long-phage) |
| `neuralbioinfo/prokbert-mini-c-phage` | 1 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-c-phage) |

### Datasets

| Dataset Name | Hugging Face URL |
| --- | --- |
| `neuralbioinfo/ESKAPE-genomic-features` | [Link](https://huggingface.co/datasets/neuralbioinfo/ESKAPE-genomic-features) |
| `neuralbioinfo/phage-test-10k` | [Link](https://huggingface.co/datasets/neuralbioinfo/phage-test-10k) |
| `neuralbioinfo/bacterial_promoters` | [Link](https://huggingface.co/datasets/neuralbioinfo/bacterial_promoters) |
| `neuralbioinfo/ESKAPE-masking` | [Link](https://huggingface.co/datasets/neuralbioinfo/ESKAPE-masking) |



# Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{ProkBERT2024,
  author  = {Ligeti, Balázs and Szepesi-Nagy, István and Bodnár, Babett and Ligeti-Nagy, Noémi and Juhász, János},
  journal = {Frontiers in Microbiology},
  title   = {{ProkBERT} family: genomic language models for microbiome applications},
  year    = {2024},
  volume  = {14},
  URL={https://www.frontiersin.org/articles/10.3389/fmicb.2023.1331233},       
	DOI={10.3389/fmicb.2023.1331233},      
	ISSN={1664-302X}
}
```
