#!/bin/bash
#
# file: run.sh
# This bash script first calls "setup.sh" to compile your code, then calls "prel#iminary_challenge.py" to generate "answers.txt".

# echo "============ running setup script ============"

# bash setup.sh

echo "==== running entry script on the test set ===="
# Clear previous answers.txt
rm -f answers.txt
# Generate new answers.txt
python preliminary_challenge.py
echo "=================== Done! ===================="


