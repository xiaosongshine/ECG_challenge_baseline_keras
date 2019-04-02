import sys
import os
import numpy as np
import scipy.io as sio
import random
from decimal import Decimal

# Usage: python preliminary_challenge.py test_file_path
def main():
    ## Add your codes to  classify normal and illness.




    ##  Classify the samples of the test set and write the results into answers.txt,
    ##  and each row representing a prediction of one sample.
    ##  Here we use random numbers as prediction labels as an example and
    ##  you should replace it with your own results.

    test_set = os.getcwd()+'/TEST'
    f_w = open('answers.txt', 'w')
    for root, subdirs, files in os.walk(test_set):
        if files:
            for records in files:

                if records.endswith('.mat'):
                    line = records.strip('.mat') + ' ' + str(int(random.randint(0, 1)))
                    f_w.write(line + '\n')




if __name__ == "__main__":
    main()
