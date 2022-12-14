from glob import glob
from tqdm import tqdm
import argparse
import json
import ndjson
from os import path as osp

parser = argparse.ArgumentParser()
parser.add_argument('ped_ann', help='ex) ~/annotation/v221130/nuimages_ped_1017/v1.0-mini/json', type=str)
parser.add_argument('obj_dataset', help='ex) ~/work/Watch-BBox-In-Out/object-extract/output/nuImages/v1.0-mini', type=str)
args = parser.parse_args()

def read_ndjson(ndjson_path):
    with open(ndjson_path, 'r') as f:
        ndj = ndjson.load(f)
        tmp = json.dumps(ndj)
        data = json.loads(tmp)
    return data

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

for ped_ann in tqdm(glob(args.ped_ann + '/*.json')):
    record_id = osp.basename(ped_ann)  # -> 19b80e9ee6d647ef8ad466960f06ffeb.json
    peds = read_ndjson(ped_ann)
    ped_list = []
    for ped in peds:
        ped_list.append(ped['token'])
    try:
        objs = read_json(args.obj_dataset + '/' + record_id)
    except FileNotFoundError as e:
        print(e, record_id)
        continue
    obj_list = objs.keys()
    if set(ped_list).issubset(obj_list):
        continue
    # obj_listに含まれない歩行者がいる場合
    for ped in ped_list:
        if ped in obj_list:
            continue
        print(f"{ped} is not found.({record_id})")