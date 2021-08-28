import os
import json
import pandas as pd

'''
    Program:
            This porgram is for looking up source of best prediction example 
    Histroy:
            2021/08/28 Eric Chen First release
    Reference:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
            https://www.datasciencelearner.com/convert-python-dict-to-csv-implementation/
'''
WLASL_INFO_PATH = '/home/Datasets/WLASL_v0.3.json'
WLASL_CSV_OUTPUT = '/home/Datasets/ASL.csv'

def parse_ASLjson(json_loaded):
    json2csv = []
    for i in  range(len(json_loaded)):
        instances = json_loaded[i]['instances']
        for instance in instances:
            result = {}
            result['video_id'] = instance['video_id']
            result['gloss'] = json_loaded[i]['gloss']
            result['source'] = instance['source']
            result['signer_id'] = instance['signer_id']
            result['bbox'] = instance['bbox']
            result['fps'] = instance['fps']
            result['frame_start'] = instance['frame_start']
            result['frame_end'] = instance['frame_end']
            result['url'] = instance['url']
            result['split'] = instance['split']
            result['variation_id'] = instance['variation_id']
            result['instance_id'] = instance['instance_id']
            json2csv.append(result)
    return json2csv

def get_wlasl(json_path=WLASL_INFO_PATH):
    with open(json_path,'r') as f:
        json_loaded = json.load(f)
    return parse_ASLjson(json_loaded= json_loaded)

def main():
    wlasl_json = get_wlasl(WLASL_INFO_PATH)
    df = pd.DataFrame.from_dict(wlasl_json)
    df.to_csv(path_or_buf = WLASL_CSV_OUTPUT, index = False, header=True)

if __name__ == '__main__':
    main()