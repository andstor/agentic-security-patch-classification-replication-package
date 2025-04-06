from datasets import load_dataset
import re

def find_cve_ids(text):
    """
    Finds all CVE ID references in a given text.
    
    Args:
        text (str): The input string to search for CVE IDs.
    
    Returns:
        list: A list of found CVE IDs.
    """
    cve_pattern = r'CVE-\d{4}-\d{4,}'
    return re.findall(cve_pattern, text)

ddict = load_dataset("fals3/cvec")

explicit_dataset = ddict.filter(lambda x: find_cve_ids(x['commit_message']), num_proc=10)

ddict.push_to_hub("fals3/cvec", config_name="explicit", private=False, max_shard_size="250MB")
