import os
import json
import pandas as pd
from WLASL2csv import 
'''
    Program:
            This porgram is for looking up source of best prediction example 
    Histroy:
            2021/08/28 Eric Chen First release
'''
WLASL_INFO_PATH = '/home/Datasets/WLASL_v0.3.json'
best_sample = ['00853', '01460', '04040', '04849', '05629']
good_sample = ['01962', '07245', '04600', '09155', '09468', '09924', '24954']



def main():
    # get WLASL information
    wlasl_json = get_wlasl(json_path = WLASL_INFO_PATH)
    print(wlasl_json)
    # video source
    ## find video id and lookup source
    # video pickup 

if __name__ == '__main__':
    main()