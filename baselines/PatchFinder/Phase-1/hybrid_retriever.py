import polars as pl
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, load_dataset



DATA_DIR = '../../../data/baselines/PatchFinder'



def grid_search(df):
    """
    Perform a grid search over the given parameter grid.
    """

    lambda_values = np.arange(0.1, 10.05, 0.5)

    best_lambda = None
    best_metric = -float("inf")  # higher is better for Hit@10 etc.

    # Store per-lambda metrics
    results = []

    total_cves = df.select(pl.col("cve")).unique().height

    for lambda_value in lambda_values:
        # Compute fused score
        scored_df = df.with_columns(
            (pl.col("similarity") + (lambda_value * pl.col("f1"))).alias("fused_f1")
        )
        
        # Sort and rank within each CVE
        ranked = (
            scored_df
            .sort(["cve", "fused_f1"], descending=[False, True])
            .with_columns([
                pl.col("fused_f1").rank("ordinal").over("cve").alias("rank")  # rank 1 = best
            ])
            .filter(pl.col("rank") <= 100)  # Top 100 per CVE
        )
        
        # Evaluate: how many CVEs have label=1 in top 100
        hit_at_100 = (
            ranked
            .filter(pl.col("label") == 1)
            .select(pl.col("cve"))
            .unique()
            .height
        )
        
        hit_ratio = hit_at_100 / total_cves
        results.append((lambda_value, hit_ratio))
        
        if hit_ratio > best_metric:
            best_metric = hit_ratio
            best_lambda = lambda_value

    # After collecting all (lambda, hit_ratio) pairs
    # Find all lambda values with the best metric
    best_metric = max(hit_ratios for _, hit_ratios in results)
    candidates = [l for l, r in results if r == best_metric]

    # Select the lambda closest to 1 among them
    best_lambda = min(candidates, key=lambda x: abs(x - 1))
    print(f"Best λ: {best_lambda:.2f}, Hit@100 Ratio: {best_metric:.4f}")
    return best_lambda


def main():
    
    
    λ = 1# grid_search(df)


    ds_cve = load_from_disk("tmp/ds_cve")
    ds_patches = load_from_disk("tmp/ds_patches")
    ds_nonpatches = load_from_disk("tmp/ds_nonpatches")
    
    for split in tqdm(['train', 'test', 'validation'], desc="Processing splits", dynamic_ncols=True):
        cindex = {key: idx for idx, key in tqdm(enumerate(ds_cve[split]["cve"]), total=len(ds_cve[split]["cve"]), desc=f"Indexing CVEs {split}")}
        pindex = {key: idx for idx, key in tqdm(enumerate(ds_patches[split]["commit_id"]), total=len(ds_patches[split]["commit_id"]), desc=f"Indexing patch commits {split}")}
        npindex = {key: idx for idx, key in tqdm(enumerate(ds_nonpatches[split]["commit_id"]), total=len(ds_nonpatches[split]["commit_id"]), desc=f"Indexing non-patch commits {split}")}
        
        
        lexical_similarity_df = pl.read_csv(os.path.join(DATA_DIR, f'lexical_similarity_{split}.csv'))
        semantic_similarity_df = pl.read_csv(os.path.join(DATA_DIR, f'semantic_similarity_{split}.csv'))

        merged_df = lexical_similarity_df.join(
            semantic_similarity_df,
            on=['commit_id', 'cve' ,'repo', 'label'],
            how='inner',
        )

        fused_df = merged_df.with_columns(
            (pl.col('similarity') + (λ * pl.col('f1'))).alias('fused_f1'),
        )

        ranked_df = (
            fused_df
            .with_columns([
                (-pl.col("fused_f1")).rank(method="ordinal").over("cve").alias("rank")  # rank 1 = best
            ])
            
            .sort(["cve", "fused_f1"], descending=[False, False])
        )

        # Save data for metrics
        ranked_df.select(['cve', 'repo', 'commit_id', 'similarity', 'label', 'recall', 'precision', 'f1', 'fused_f1']).to_pandas().to_csv(os.path.join(DATA_DIR, f'hybrid_similarity_{split}.csv'))

        
        ranked_top100_df = (
            ranked_df
            .filter(pl.col("rank") <= 100)  # Top 100 per CVE
        )
        
        hit_at_100 = (
            ranked_top100_df
            .filter(pl.col("label") == 1)
            .select(pl.col("cve"))
            .unique()
            .height
        )
        
        print(f"Hit@100: {hit_at_100} / {merged_df.select(pl.col('cve')).unique().height} = {hit_at_100 / merged_df.select(pl.col('cve')).unique().height:.4f}")


        # Create and write the header of the CSV file

        empty_df = pd.DataFrame(columns=['cve', 'repo', 'commit_id', 'similarity', 'label', 'desc_token', 'msg_token', 'diff_token', 'recall', 'precision', 'f1', 'fused_f1'])
        empty_df.to_csv(os.path.join(DATA_DIR, f'top100_{split}.csv'), index=False)



        def assemble_row(example):
            cve = example["cve"]
            commit_id = example["commit_id"]
            
            cve_row = ds_cve[split][cindex[cve]]
            
            commit_row = None
            if commit_id in pindex: # Patch commit
                commit_row = ds_patches[split][pindex[commit_id]]
            else: # Non-patch commit
                commit_row = ds_nonpatches[split][npindex[commit_id]]
            
            if commit_row is not None:
                return {
                    "cve": cve,
                    "repo": commit_row["repo"],
                    "commit_id": commit_id,
                    "similarity": example["similarity"],
                    "label": example["label"],
                    "desc_token": cve_row["desc_token"],
                    "msg_token": commit_row["msg_token"],
                    "diff_token": commit_row["diff_token"],
                    "recall": example["recall"],
                    "precision": example["precision"],
                    "f1": example["f1"],
                    "fused_f1": example["fused_f1"]
                }
            else:
                return None

        #'cve', 'repo', 'commit_id', 'similarity', 'label', 'recall', 'precision', 'f1', 'fused_f1'
        for cve, group_df in tqdm(ranked_top100_df.group_by("cve"), desc="Processing CVE groups", total=ranked_top100_df.select(pl.col("cve")).unique().height, dynamic_ncols=True):
            
            cve_df = group_df.apply(assemble_row, axis=1, result_type='expand')
            
            file_path = os.path.join(DATA_DIR, f'top100_{split}.csv')
            cve_df.to_csv(file_path, mode='a', header=False, index=False)
        print(f"Data written to {file_path} for split {split}")
        




if __name__ == '__main__':
    # Load the data
    main()