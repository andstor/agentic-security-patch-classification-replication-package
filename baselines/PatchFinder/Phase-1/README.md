# About this folder

## Folder structure
```bash
tmp                         ## directory for temporary files
README.md
preprocess_data.py          ## for data preparation only, to preprocess the dataset
lexical_similarity.py       ## for lexical-based retriever
semantic_similarity.py      ## for semantic-based retriever
hybrid_retriever.py         ## for hybrid retriever
```

## Phase 1
> This is the first phase of the PatchFinder project, which focuses on lexical and semantic similarity-based retrieval methods. The goal is to get top-100 results for the given dataset.

To run it, we assume that you have prepared the evaluation data, and now we calculate the tf-idf scores, then rank them.

1. Run `preprocess_data.ipynb` first to preprocess the dataset.
2. Run `lexical_similarity.py` to get the tf-idf scores.
3. Run `semantic_similarity.py` to get the semantic similarity scores.
4. Run `hybrid_retriever.py` to rank the commits, and get the top-100 results. The results will be saved into `top_100_fusion.csv` by default.
