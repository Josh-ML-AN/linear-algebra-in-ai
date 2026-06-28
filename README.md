# Linear Algebra in AI: Recommender Walkthrough

This repository contains the companion notebook for Part 2 (with references to Part 1) of a two-part series on linear algebra in AI.

The notebook builds a lightweight recommendation model using the MovieLens 32M dataset. The goal is to make the core linear algebra ideas behind modern AI systems easier to inspect.

The walkthrough covers:

* sparse user-movie ratings matrices
* mean-centered ratings
* cosine similarity
* singular value decomposition
* dense movie embeddings
* attention-style weighting
* context-based recommendations
* validation against simple baselines
* the connection between recommender systems and transformer-based language models

## Main Idea

The recommender follows the same broad pattern that appears inside attention-based AI systems:

```text
represent information as vectors
compare those vectors
weight the relevant signals
combine them into context
use that context to make a prediction
```

In this notebook, the objects are movies and ratings.

In a transformer-based language model, the objects are tokens and learned embeddings.

The scale and implementation are different, but the underlying linear algebra pattern is similar.

## Repository Structure

```text
.
├── linear_algebra_exploration_companion_final.ipynb
├── README.md
├── requirements.txt
└── ml-32m/
    ├── movies.csv
    └── ratings.csv
```

The `ml-32m/` folder is not included in this repository. Download the MovieLens 32M dataset separately and place the required files in that folder.

## Data Setup

Download the MovieLens 32M dataset from GroupLens.

After unzipping the dataset, place these files in a folder called `ml-32m`:

```text
ml-32m/
├── movies.csv
└── ratings.csv
```

The notebook expects this folder to sit in the root of the repository.

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open the notebook:

```text
linear_algebra_exploration_companion_final.ipynb
```

Then run the notebook from top to bottom.

## What the Notebook Does

The notebook walks through the full recommendation pipeline:

1. Load the MovieLens ratings and movie metadata.
2. Filter the data to a denser recommendation subset.
3. Build a sparse user-movie ratings matrix.
4. Mean-center the ratings.
5. Use SVD to create 32-dimensional movie embeddings.
6. Build an attention-style context recommender.
7. Generate recommendations from different user-history snapshots.
8. Validate the recommender against simple baselines.

## Validation

The recommender is evaluated using a held-out retrieval task.

For each user, a small number of highly rated movies are removed from their history. The model is then given the remaining rating history and asked to rank the full movie catalog.

The main question is:

Can the model recover the held-out liked movies near the top of the recommendation list?

The notebook compares the attention-style recommender against:

* global movie average
* movie popularity
* random ranking

This is a lightweight validation setup, not a full production recommendation benchmark. It is designed to test whether the model is using user-specific context rather than simply recommending popular movies.

## Requirements

A minimal `requirements.txt` should include:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
jupyter
ipython
```
