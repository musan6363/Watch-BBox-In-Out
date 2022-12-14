import argparse
from glob import glob
import uuid
import json
import ndjson
from os import path as osp

FRAME_SIZE = (1980, 1200)  # nuImages
N_ANNOTATE = 3
ANN_DATA_PATH = "/home0/murakamih/annotation/v221130/nuimages_ped_1017/v1.0-train/json"

class GazeTargetObject:
    def __init__(self, token: str, bbox: list, category: str) -> None:
        self.token = token
        self.bbox = bbox  # [left-top-x, left-top-y, right-bottom-x, right-bottom-y]
        self.category = category

    def is_inside(self, coord: tuple) -> bool:
        if self.bbox[0] <= coord[0] <= self.bbox[2] and self.bbox[1] <= coord[1] <= self.bbox[3]:
            return True
        else:
            return False
        
class GazePoint:
    def __init__(self, ped_token: str, frame_token: str, loc: tuple, is_difficult: bool, is_eyecontact: bool) -> None:
        self.token = uuid.uuid4().hex
        self.ped_token = ped_token
        self.frame_token = frame_token
        self.loc = loc
        self.is_difficult = is_difficult
        self.is_eyecontact = is_eyecontact
        self.is_inside_frame = True if 0 < self.loc[0] < FRAME_SIZE[0] and 0 < self.loc[1] < FRAME_SIZE[1] else False
        self.is_inside_object = False
        self.gto = []

class Pedestrian:
    def __init__(self, token: str, frame_token: str, bbox: list) -> None:
        self.token = token
        self.frame_token = frame_token
        self.bbox = bbox
        self.gaze = []
        self.gto = []

    def set_gaze_point(self, look: list, difficult: list, eyecontact: list) -> list:
        for i in N_ANNOTATE:
            gp = GazePoint(self.token, self.frame_token, tuple(look[i]), difficult[i], eyecontact[i])
            self.gaze.append(gp)
        return self.gaze

def parse_args():
    parser = argparse.ArgumentParser(description='Count what look the gaze point')
    parser.add_argument('ann_json', help='ex) /home0/murakamih/annotation/v221205/nuimages_ped_1017/v1.0-train/json')
    parser.add_argument('save_dir', help='ex) ./output')
    _args = parser.parse_args()
    return _args

def read_ndjson(path: str) -> list:
    with open(path, 'r') as f:
        _ndj = ndjson.load(f)
        _tmp = json.dumps(_ndj)
        _data = json.loads(_tmp)
    return _data

def main():
    args = parse_args()
    ann_jsons = glob(args.ann_json+"/*.json")
    dataset_name = args.ann_json.split('/')[-3]
    version_name = args.ann_json.split('/')[-2]
    for ann_json in ann_jsons:
        image_token = osp.splitext(osp.basename(ann_json))[0]
        print(dataset_name, version_name, image_token)
        records = read_ndjson(ann_json)
        for record in records:
            print(record)
        break

if __name__ == "__main__":
    main()