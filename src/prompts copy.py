USER_PROMPT_PATCH_INFO = """Evaluate if commit {commit_id} in {repository} is the correct patch for {cve_id}.

CVE ID: {cve_id}

Commit Message: "{commit_message}"

Code Changes:
```
{commit_diff}
```
"""



USER_PROMPT_TIER_1 = """
Given a Git commit message, a Git diff (code changes), and a CVE (Common Vulnerabilities and Exposures) ID, determine if the provided commit is the correct patch for the given CVE ID.
Assess if the commit message and Git diff directly address the vulnerability. If the commit message mentions the CVE ID or describes changes corresponding to the vulnerability, and/or the Git diff shows changes that clearly mitigate the vulnerability, the commit is likely the correct patch.

Example:
---
CVE ID: CVE-2023-1234

Commit Message: "Fix buffer overflow vulnerability in user input handling (CVE-2023-1234)"

Code Changes:
```
diff --git a/src/input_handler.py b/src/input_handler.py
index 17d60d3..58a226f 33188
--- a/src/input_handler.py
+++ b/src/input_handler.py
@@ -10,7 +10,7 @@
 def handle_user_input(user_input):
-    buffer = user_input
+    buffer = user_input[:1024]  # Limit input to 1024 characters
     # Process the input
     return process_input(buffer)
```
---
Above example commit message above mentions the CVE ID and describes a fix for a buffer overflow vulnerability.
The code changes limit user input to 1024 characters, directly addressing the buffer overflow issue.
The commit is therefore very likely the correct patch for CVE-2023-1234.


Provide a yes or no answer, along with an explanation of your decision.

{user_prompt_patch_info}

Based on the provided information, evaluate if the commit is the correct patch for the given CVE.
"""


USER_PROMPT_TIER_2 = """
Given a Git commit message, a Git diff (code changes), and a CVE (Common Vulnerabilities and Exposures) ID, determine if the provided commit is the correct patch for the given CVE ID.

Evaluate the commit message and vulnerability report for external references that may confirm the commit as the correct patch.
Look for links, references to GitHub issues or pull requests, or other external sources in the commit message and vulnerability report. Follow these references to determine if they:

- Directly link to the commit ID on GitHub or another version control platform.
- Describe the vulnerability and the fix in a way that matches the commit message and code changes.
- Provide additional context or confirmation that the commit is the correct patch for the CVE.

Provide a yes or no answer, along with an explanation.

{user_prompt_patch_info}

Based on the provided information, evaluate if the commit is the correct patch for the given CVE ID.
"""