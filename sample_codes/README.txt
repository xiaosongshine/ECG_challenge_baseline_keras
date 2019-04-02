The Preliminary_Sample.zip contains the following components:

* Python scripts:
   -- preliminary_challenge.py (necessary) -  add your codes to classify normal and abnormal; NOTE you need to write the results into "answers.txt", and the dataset "TEST" should be in the same folder with this script.

* BASH scripts:
   -- setup.sh (optional) - a bash script runs once before any other code from the entry; use this to compile your code as needed and to install additional packages
   -- run.sh (necessary) - a script first calls "setup.sh" to compile your code, then calls "preliminary_challenge.py" to generate "answers.txt"
     
* Other files:
   -- answers.txt (necessary) - a text file containing the results of running your program on each record in test set.

* README.txt - this file

We verify that your code is working as you intended, by running "run.sh" on the test set, then comparing the answers.txt file that you submit with your
entry with answers produced by your code running in our test environment using the same records.

