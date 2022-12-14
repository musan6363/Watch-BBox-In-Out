import argparse
from glob import glob
from tqdm import tqdm
import uuid
import json
import ndjson
from os import path as osp
import statistics
import sys

FRAME_SIZE = (1980, 1200)  # nuImages
N_ANNOTATE = 3

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
        self.is_inside_frame = self.check_inside_frame()
        self.is_inside_object = False
        self.gto = []

    def check_inside_frame(self) -> bool:
        if self.is_eyecontact:
            return False
        if 0 < self.loc[0] < FRAME_SIZE[0] and 0 < self.loc[1] < FRAME_SIZE[1]:
            return True
        return False

    def add_gto(self, obj: GazeTargetObject) -> bool:
        if self.is_eyecontact:
            return False
        if obj.token == self.ped_token:
            return False
        if obj.is_inside(self.loc):
            self.gto.append(obj)
            self.is_inside_object = True
            return True
        return False

class Pedestrian:
    def __init__(self, token: str, frame_token: str, bbox: list) -> None:
        self.token = token
        self.frame_token = frame_token
        self.bbox = bbox
        self.gaze = []
        self.gto = []

    def set_gaze_point(self, look: list, difficult: list, eyecontact: list) -> None:
        for i in range(N_ANNOTATE):
            if len(look[i]) == 0 or difficult[i] == "" or eyecontact[i] == "":
                continue
            gp = GazePoint(self.token, self.frame_token, tuple(look[i]), convert_bool_flag(difficult[i]), convert_bool_flag(eyecontact[i]))
            self.gaze.append(gp)

    def add_gto(self, gto: GazeTargetObject) -> None:
        '''
        現状はGazePoint内のgtoと変わらない
        将来的に，difficultフラグや多数決を経て，そのpedが見ている代表オブジェクトを決定した上で格納したい
        '''
        self.gto.append(gto)

class Frame:
    def __init__(self, token: str) -> None:
        self.token = token
        self.peds = []
        self.objects = []

        # 統計情報
        self.looking_eyecontact = 0
        self.looking_inside_frame = 0
        self.looking_inside_obj = 0
    
    def add_ped(self, ped: Pedestrian) -> None:
        self.peds.append(ped)

    def add_obj(self, obj: GazeTargetObject) -> None:
        self.objects.append(obj)

    def stat(self) -> None:
        ped: Pedestrian
        for ped in self.peds:
            gaze: GazePoint
            for gaze in ped.gaze:
                if gaze.is_eyecontact:
                    self.looking_eyecontact += 1
                    continue
                if gaze.is_inside_frame:
                    self.looking_inside_frame += 1
                if gaze.is_inside_object:
                    self.looking_inside_obj += 1

def convert_bool_flag(b: str) -> bool:
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError(b)

def read_ndjson(path: str) -> list:
    with open(path, 'r') as f:
        _ndj = ndjson.load(f)
        _tmp = json.dumps(_ndj)
        _data = json.loads(_tmp)
    return _data

def read_json(path: str) -> list:
    with open(path, 'r') as f:
        _data = json.load(f)
    return _data
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ped_ann', help='ex) ~/annotation/v221130/nuimages_ped_1017/v1.0-mini/json', type=str)
    parser.add_argument('obj_dataset', help='ex) ~/work/Watch-BBox-In-Out/object-extract/output/nuImages/v1.0-mini', type=str)
    parser.add_argument('--not_use', help='ex) ~/annotation/v221130/nuimages_ped_1017/v1.0-train/notuse.json', type=str)
    args = parser.parse_args()
    return args

def check_looking(peds: list, obj: GazeTargetObject) -> None:
    ped: Pedestrian
    for ped in peds:
        gp: GazePoint
        for gp in ped.gaze:
            is_looking = gp.add_gto(obj)  # 各アノテーション点がobjを見ているか調査．見ていればリストに追加してTrueを返す．
            if is_looking:
                ped.add_gto(obj)


def main():
    args = parse_args()


    not_use_frame = {}
    if args.not_use:
        not_use = read_json(args.not_use)
        for nu in not_use:
            frame = nu['img_token']
            if not frame in not_use_frame.keys():
                not_use_frame[frame] = []
            not_use_frame[frame].append(nu['ped_token'])

    frames = {}
    for ped_record in tqdm(glob(args.ped_ann + '/*.json'), "running record..."):
        frame_id = osp.splitext(osp.basename(ped_record))[0]
        frame = Frame(frame_id)

        # not useレコードの照会
        not_use_ped = []
        if frame_id in not_use_frame.keys():
            not_use_ped = not_use_frame[frame_id]

        # 歩行者の登録
        ped_annotations: list = read_ndjson(ped_record)
        ped_annotation: dict
        for ped_annotation in ped_annotations:
            if ped_annotation['token'] in not_use_ped:
                continue
            ped = Pedestrian(ped_annotation['token'], frame_id, ped_annotation['bbox'])
            ped.set_gaze_point(ped_annotation['look'], ped_annotation['difficult'], ped_annotation['eyecontact'])
            frame.add_ped(ped)
        # オブジェクトの登録
        dataset_objects: dict = read_json(args.obj_dataset + '/' + frame_id + '.json')
        object_token: str
        dataset_object: dict
        for object_token, dataset_object in dataset_objects.items():
            obj = GazeTargetObject(object_token, dataset_object['bbox'], dataset_object['category'])
            frame.add_obj(obj)
            check_looking(frame.peds, obj)  # オブジェクトを各視点が見ているか確認し，格納

        # frameごとの統計情報
        frame.stat()
        frames[frame_id] = frame

    # 統計情報の表示
    cnt_peds = []
    cnt_objs = []
    cnt_looking_eyecontact = []
    cnt_looking_inside_frame = []
    cnt_looking_inside_obj = []
    cnt_gaze_points = 0
    frame: Frame
    for frame in tqdm(frames.values(), "statistics..."):
        cnt_peds.append(len(frame.peds))
        cnt_objs.append(len(frame.objects))
        cnt_looking_eyecontact.append(frame.looking_eyecontact)
        cnt_looking_inside_frame.append(frame.looking_inside_frame)
        cnt_looking_inside_obj.append(frame.looking_inside_obj)
        ped: Pedestrian
        for ped in frame.peds:
            cnt_gaze_points += len(ped.gaze)
    print(f"歩行者総数 -> {sum(cnt_peds)}")
    if len(cnt_peds) > 1:
        print(f"フレームあたりの歩行者数 -> {statistics.mean(cnt_peds):.02f} \u00B1 {statistics.stdev(cnt_peds):.02f}")
    print(f"オブジェクト総数 -> {sum(cnt_objs)}")
    if len(cnt_objs) > 1:
        print(f"フレームあたりのオブジェクト数 -> {statistics.mean(cnt_objs):.02f} \u00B1 {statistics.stdev(cnt_objs):.02f}")
    print(f"総アノテーション点数 -> {cnt_gaze_points}({(cnt_gaze_points*100/(sum(cnt_peds)*3)):.02f}%)")
    print(f"アイコンタクト点数 -> {sum(cnt_looking_eyecontact)}({(sum(cnt_looking_eyecontact)*100/cnt_gaze_points):.02f}%)")
    print(f"フレーム内を見ている点数 -> {sum(cnt_looking_inside_frame)}({(sum(cnt_looking_inside_frame)*100/cnt_gaze_points):.02f}%)")
    print(f"オブジェクト内を見ている点数 -> {sum(cnt_looking_inside_obj)}({(sum(cnt_looking_inside_obj)*100/cnt_gaze_points):.02f}%)")

if __name__ == "__main__":
    main()