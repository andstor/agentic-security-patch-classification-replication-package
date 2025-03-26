import json
from datasets import Dataset, Features, Value
from ftfy import fix_encoding
from datasets import disable_caching


def jsonl_generator():
    file_path = "./output/commits_data.jsonl"
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

#disable_caching()

def infer_features(file_path):
    """Infer dataset features from the first record of the JSONL file."""
    with open(file_path, 'r') as f:
        first_record = json.loads(f.readline().strip())
    
    
    # Infer feature types based on the first record
    features = {}
    for key, value in first_record.items():
        if isinstance(value, int):
            features[key] = Value('int32')
        elif isinstance(value, float):
            features[key] = Value('float32')
        elif isinstance(value, str):
            features[key] = Value('string')
        # Add more types if necessary, e.g., for dates or booleans.
        else:
            features[key] = Value('string')  # Default to string for unknown types
    return Features(features)


FILE_PATH = "./output/commits_data.jsonl"
def main():
    # Infer the features and types from the first record
    features = infer_features(FILE_PATH)
    # Create a generator for the JSONL data
    #gen = jsonl_generator(file_path)   
    dataset = Dataset.from_generator(generator=jsonl_generator, features=features, keep_in_memory=True)# TypeError: cannot pickle 'generator' object

    dataset.push_to_hub("andstor/cvevc", config_name="raw", private=True, max_shard_size="250MB")


if __name__ == "__main__":
    main()
