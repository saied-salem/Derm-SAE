#!/bin/bash
# Runs the vocabulary preparation script.
set -e
echo "--- 1/6: Preparing Vocabulary ---"

# This assumes 01_prepare_vocabulary.py is in the same directory
python 01_prepare_vocabulary.py

echo "--- Step 1 Complete ---"
