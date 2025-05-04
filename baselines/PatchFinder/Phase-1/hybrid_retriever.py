import polars as pl
import os
import numpy as np
import pandas as pd
from tqdm import tqdm



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
                pl.col("fused_f1").rank("dense").over("cve").alias("rank")  # rank 1 = best
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
    
    
    best_lambda = 1# grid_search(df)


# read file
    split = 'train'
    
    for split in tqdm(['train', 'test', 'validation'], desc="Processing splits", dynamic_ncols=True):
        
        lexical_similarity_df = pl.read_csv(os.path.join(DATA_DIR, f'lexical_similarity_{split}.csv'))
        semantic_similarity_df = pl.read_csv(os.path.join(DATA_DIR, f'semantic_similarity_{split}.csv'))

        merged_df = lexical_similarity_df.join(
            semantic_similarity_df,
            on=['commit_id','cve','owner','repo', 'label'],
            how='inner',
        )

        λ = best_lambda
        fused_df = merged_df.with_columns(
            (pl.col('similarity') + (λ * pl.col('f1'))).alias('fused_f1'),
        )

        ranked_df = (
            fused_df
            .sort(["cve", "fused_f1"], descending=[False, True])
            .with_columns([
                pl.col("fused_f1").rank("dense").over("cve").alias("rank")  # rank 1 = best
            ])
            .filter(pl.col("rank") <= 100)  # Top 10 per CVE
        )

        hit_at_100 = (
            ranked_df
            .filter(pl.col("label") == 1)
            .select(pl.col("cve"))
            .unique()
            .height
        )
        
        print(f"Hit@100: {hit_at_100} / {merged_df.select(pl.col('cve')).unique().height}")


        # Create and write the header of the CSV file

        empty_df = pd.DataFrame(columns=['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label', 'desc_token', 'msg_token', 'diff_token', 'recall', 'precision', 'f1', 'fused_f1'])
        empty_df.to_csv(os.path.join(DATA_DIR, f'top_100_fusion_{split}.csv'), index=False)


        cve_path = f"tmp/tokenized/cve_{split}.parquet"
        patches_path = f"tmp/tokenized/patches_{split}.parquet"
        nonpatches_path = f"tmp/owner_repo_groups/{split}"

        cve_df = pl.scan_parquet(cve_path)
        patches_df = pl.scan_parquet(patches_path)

        for cve, df in  tqdm(ranked_df.group_by("cve"), desc="Processing CVE groups", total=ranked_df.select(pl.col("cve")).unique().height, dynamic_ncols=True):
            owner_repo_name = ranked_df[0]['owner'].item() + "_" + ranked_df[0]['repo'].item() # File name key
            nonpatches_df = pl.scan_parquet(os.path.join(nonpatches_path, f"{owner_repo_name}.parquet")).collect()
            commits_df = pl.concat([patches_df, nonpatches_df], how="vertical")
            data = (
                df
                .join(commits_df, on=['commit_id'], how='left')
                .select(['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label', 'desc_token', 'msg_token', 'diff_token', 'recall', 'precision', 'f1', 'fused_f1'])
            )

            file_path = os.path.join(DATA_DIR, f'semantic_similarity_{split}.csv')
            data.to_pandas().to_csv(file_path, mode='a', header=False, index=False)
        print(f"Data written to {file_path} for split {split}")
        




if __name__ == '__main__':
    # Load the data
    main()