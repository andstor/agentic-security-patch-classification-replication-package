https://github.com/ntgiang71096/vfdetector/blob/main/vulfixminer.py



1. Phase 1: Fine-tuning

- Extract file-level code changes
  - Extract rem-code and add-code from from each file

  extract file-level diffs.

- Process the removed code and added code lines
  - pip install codeprep

- Input tokens construction
  - CLS, SEP, EOS tokens. all same length (pad or truncate)

2. Phase 2: Training