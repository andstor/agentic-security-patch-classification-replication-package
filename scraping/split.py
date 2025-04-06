import json
from datasets import Dataset, Features, Value, concatenate_datasets, DatasetDict
from datasets import Features, Value
from functools import partial
from tqdm import tqdm

def jsonl_generator(file_path):
    """Yield lines from a JSONL file as dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # ensure utf-8 encoding
            try:
                #encode commit message to utf-8
                #data['diff'] = data['diff'].encode('utf-8', 'surrogateescape').decode('ISO-8859-1')
                data['diff'] = data['diff'].encode('utf-8', "replace").decode('utf-8')
                yield data
            except:
                #print(f"Error encoding line: {line}")
                raise ValueError(f"Error encoding line")


def split_numbers_from_dict(data, percent_train=0.80, percent_test=0.20):
    # Ensure that the sum of the percentages equals 1
    if percent_train + percent_test != 1:
        raise ValueError("The sum of the percentages must equal 1 (e.g., 0.80 + 0.20).")
    
    # Calculate the total sum of the values (weights)
    total_sum = sum(data.values())
    
    # Calculate target sums for each group based on adjustable percentages
    target_train = total_sum * percent_train
    target_test = total_sum * percent_test
    
    # Sort the dictionary by value (weight) in descending order
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    
    train, test = {}, {}
    sum_train, sum_test = 0, 0
    
    # Keep track of whether the last item went into train or test to ensure balance
    for key, weight in sorted_items:
        # Try to balance the distribution across both sets
        if sum_train + weight <= target_train:
            train[key] = weight
            sum_train += weight
        elif sum_test + weight <= target_test:
            test[key] = weight
            sum_test += weight
        else:
            # In case one side is fully filled, add the item to the side with less sum
            if sum_train < sum_test:
                train[key] = weight
                sum_train += weight
            else:
                test[key] = weight
                sum_test += weight

    return train, test


# Load the dataset
features = Features({
    'commit_id': Value(dtype='string'),
    'owner': Value(dtype='string'),
    'repo': Value(dtype='string'),
    'commit_message': Value(dtype='string'),
    'diff': Value(dtype='string')
})


dataset_commits = Dataset.from_generator(generator=partial(jsonl_generator, "./output/commits_data.jsonl"), features=features)
new_column = [0] * len(dataset_commits)
dataset_commits = dataset_commits.add_column("label", new_column)
print(f"Number of non-patching commits: {len(dataset_commits)}")

dataset_patches = Dataset.from_generator(generator=partial(jsonl_generator, "./output/patches_data.jsonl"), features=features)
new_column = [1] * len(dataset_patches)
dataset_patches = dataset_patches.add_column("label", new_column)
print(f"Number of patcing commits: {len(dataset_patches)}")

dataset = concatenate_datasets([dataset_commits, dataset_patches], axis=0)

# sort the dataset by owner and repo
dataset = dataset.sort(column_names=['owner', 'repo'], reverse=False)

repos = {} # "repo_owner/repo_name" : count
for commit in tqdm(dataset):
    repo_ident = commit['owner'] + "/" + commit['repo']
    if repo_ident not in repos:
        repos[repo_ident] = 0
    repos[repo_ident] += 1


train_index, test_index = split_numbers_from_dict(repos, percent_train=0.80, percent_test=0.20)
test_index, validation_index = split_numbers_from_dict(test_index, percent_train=0.50, percent_test=0.50)

train_dataset = dataset.filter(lambda x: (x["owner"] + "/" + x["repo"]) in train_index.keys(), num_proc=10)
test_dataset = dataset.filter(lambda x: (x["owner"] + "/" + x["repo"]) in test_index.keys(), num_proc=10)
val_dataset = dataset.filter(lambda x: (x["owner"] + "/" + x["repo"]) in validation_index.keys(), num_proc=10)


ddict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "validation": val_dataset,
})

ddict.push_to_hub("fals3/cvec", config_name="default", private=False, max_shard_size="250MB")


