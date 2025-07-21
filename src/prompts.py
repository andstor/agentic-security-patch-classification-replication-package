USER_PROMPT_PATCH_INFO = """CVE ID: {cve_id}

Commit ID: {commit_id}
Repository: {repository}

Commit Message: "{commit_message}"

Code Changes:
```diff
{commit_diff}
```
"""



USER_PROMPT_TIER_1 = """
Given a Git commit message, a Git diff (code changes), and a CVE (Common Vulnerabilities and Exposures) ID, determine if the provided commit is the correct patch for the given CVE.
Assess if the commit message and/or Git diff address the vulnerability. For example, the commit message might mention the CVE ID or describes changes closely mathing the vulnerability description. Further, the Git diff might show changes that clearly mitigate the vulnerability. These are good indicators for the commit being the correct patch.

{user_prompt_patch_info}

Based on the provided information, evaluate if the commit is the correct patch for the given CVE.
Provide a True or False answer.
"""


USER_PROMPT_TIER_2 = """
Given a Git commit message, a Git diff (code changes), and a CVE (Common Vulnerabilities and Exposures) ID, determine if the provided commit is the correct patch for the given CVE.

Evaluate the commit message and vulnerability report for external references that may confirm the commit as the correct patch.
Look for links, references to GitHub issues or pull requests, or other external sources in the commit message and vulnerability report. Follow these references to determine if they:

- Directly link to the commit ID on GitHub or another version control platform.
- Describe the vulnerability and the fix in a way that matches the commit message and code changes.
- Provide additional context or confirmation that the commit is the correct patch for the CVE.

{user_prompt_patch_info}

Based on the provided information, evaluate if the commit is the correct patch for the given CVE.
Provide a True or False.
"""

USER_PROMPT_TIER_3 = """
Given a Git commit message, a Git diff (code changes), and a CVE (Common Vulnerabilities and Exposures) ID, determine if the provided commit is the correct patch for the given CVE.

Analyze the code changes in the Git diff to determine if they address the vulnerability described in the CVE. You may need to understand the vulnerability description and the code changes to make an informed decision.
This will most likely involve tracing the changes in the commit diff through the codebase to understand how they mitigate the vulnerability. Please search for files and code snippets that are relevant to the vulnerability and the code changes in the commit. However, you can NOT ask for git history or specific commits. You can only look at the current state of the repository.

{user_prompt_patch_info}

Based on the provided information, evaluate if the commit is the correct patch for the given CVE. Whatever conclusionyou arrive at, please verify the vulnerability is indeed solved by looking at the related files (not just the diff!). Ask the file_navigation developer for help.
Provide a True or False.
"""

USER_PROMPT_1 = """
Your task is to locate the security patching commit for the CVE (Common Vulnerabilities and Exposures) ID `{cve_id}`.

To solve this task you must do the following:

1. Analyze the CVE report to understand the vulnerability. Here you might find clues, such as the affected code, the nature of the vulnerability, and the potential impact. one tip is to look for links to the code repository or the patching commit.

2. Based on the collected CVE information, you will need to inspect the codebase to find the part of the code that addresses the vulnerability. If you find the vulnerability still exists, the task is not possible to solve.

3. Once you have identified the lines of code that are responsible for patching the vulnerability, you will need to find the commit that introduced these changes. You should look at the line blame to find the commit that introduced the vulnerability.

You will be provided a Git repository checked out at the time of the vulnerability report. You can use the tools provided to help you navigate the repository and find the necessary information.
"""

USER_PROMPT_LOCATION = """
Your task is to locate the code line(s) that addresses the CVE (Common Vulnerabilities and Exposures) ID `{cve_id}`. Assume that the vulnerability has been patched in the codebase.
"""

USER_PROMPT_LOCATIONOLD = """
Your task is to locate the security patching commit for the CVE (Common Vulnerabilities and Exposures) ID `{cve_id}`. Assume that the vulnerability has been patched in the codebase. Find the code location that addresses the vulnerability.
"""

USER_PROMPT_ORCHESTRATOR = """
Given a Git commit message, a Git diff (code changes), and a CVE (Common Vulnerabilities and Exposures) ID, determine if the provided commit is the correct patch for the given CVE.
You have tier1, tier2, and tier3 experts available to help you evaluate the commit. Tier1 experts can provide a quick assessment based on the commit message and diff. Tier2 experts can investigate external references in the commit message and vulnerability report. Tier3 experts can analyze the code changes in the Git diff to determine if they address the vulnerability. You can choose which experts to consult based on the complexity of the evaluation required. However, be as efficient as possible in your selection to ensure timely resolution. Tier1 experts are the fastest but may not provide the most in-depth analysis. Tier 2 experts take longer but can provide additional context from external references. Tier3 experts take the longest but provide the most detailed analysis of the code changes.

{user_prompt_patch_info}

Based on the provided information, evaluate if the commit is the correct patch for the given CVE.
Provide a True or False.
"""