import argparse
from glob import glob
from tqdm import tqdm
import uuid
import json
import ndjson
from os import path as osp
import os
import statistics
import sys

FRAME_SIZE = (1980, 1200)  # nuImages
# FRAME_SIZE = (1920*3, 1280)  # Waymo
N_ANNOTATE = 3

class GazeTargetObject:
    def __init__(self, token: str, bbox: list, category: str) -> None:
        self.token = token
        self.bbox = bbox  # [left-top-x, left-top-y, right-bottom-x, right-bottom-y]
        self.category = category
        self.area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        self.center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]

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
        self.gto = None
        self.gto_candidate = []
        self.looking_outside_cnt = 0

    def set_gaze_point(self, look: list, difficult: list, eyecontact: list) -> None:
        for i in range(N_ANNOTATE):
            if len(look[i]) == 0 or difficult[i] == "" or eyecontact[i] == "":
                continue
            gp = GazePoint(self.token, self.frame_token, tuple(look[i]), convert_bool_flag(difficult[i]), convert_bool_flag(eyecontact[i]))
            self.gaze.append(gp)
            self.looking_outside_cnt += 1 if not (gp.is_eyecontact or gp.is_inside_frame) else 0

    def add_gto(self, gto: GazeTargetObject) -> None:
        '''
        2??????????????????????????????????????????????????????????????????
        2??????????????????????????????????????????????????????????????????????????????????????????????????????
        '''
        self.gto_candidate.append(gto)
        _current_gto: GazeTargetObject
        _current_gto = self.gto
        if _current_gto is not None:
            if _current_gto.area < gto.area:
                return
        self.gto = gto

class Frame:
    def __init__(self, token: str) -> None:
        self.token = token
        self.peds = []
        self.objects = []

        # ????????????
        self.looking_eyecontact = 0
        self.looking_difficult = 0
        self.looking_inside_frame = 0
        self.looking_inside_obj = 0
        self.looking_multi_obj = []
    
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
                if gaze.is_difficult:
                    self.looking_difficult += 1
                if gaze.is_inside_frame:
                    self.looking_inside_frame += 1
                if gaze.is_inside_object:
                    self.looking_inside_obj += 1
            if len(ped.gto_candidate) > 1:
                self.looking_multi_obj.append(ped.token)


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

def export_json(dst: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(dst, f, indent=2)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ped_ann', help='ex) ~/annotation/v221130/nuimages_ped_1017/v1.0-mini/json', type=str)
    parser.add_argument('obj_dataset', help='ex) ~/work/Watch-BBox-In-Out/object-extract/output/nuImages/v1.0-mini', type=str)
    parser.add_argument('--not_use', help='ex) ~/annotation/v221130/nuimages_ped_1017/v1.0-train/notuse.json', type=str)
    parser.add_argument('--output', help='ex) output/nuimages_ped_1017/v1.0-mini', type=str, default='output')
    args = parser.parse_args()
    return args

def check_looking(peds: list, obj: GazeTargetObject) -> tuple:
    dupulicated_double = 0
    dupulicated_triple = 0
    ped: Pedestrian
    for ped in peds:
        tmp_cnt_dupulicated = 0
        gp: GazePoint
        for gp in ped.gaze:
            is_looking = gp.add_gto(obj)  # ??????????????????????????????obj??????????????????????????????????????????????????????????????????True????????????
            if is_looking:
                tmp_cnt_dupulicated += 1
        if tmp_cnt_dupulicated >= 2:
            ped.add_gto(obj)
        if tmp_cnt_dupulicated == 2:
            dupulicated_double += 1
        elif tmp_cnt_dupulicated == 3:
            dupulicated_triple += 1
    return dupulicated_double, dupulicated_triple

def calc_center(coord: list) -> list:
    center_coord = [0, 0]
    for c in coord:
        center_coord[0] += c[0]
        center_coord[1] += c[1]
    center_coord = [center_coord[0]/len(coord), center_coord[1]/len(coord)]
    return center_coord

def export_ped_obj_set(frames: list, output_dir: str) -> None:
    cnt_out_ped = 0
    cnt_obj_ped = 0
    frame: Frame
    for frame in tqdm(frames.values(), "export obj set"):
        dst = {}
        ped: Pedestrian
        for ped in frame.peds:
            if ped.looking_outside_cnt > 1:
                _outside_coords = []
                gp: GazePoint
                for gp in ped.gaze:
                    if gp.is_inside_frame:
                        continue
                    _outside_coords.append(gp.loc)
                outside_coord_avg = calc_center(_outside_coords)
                dst[ped.token] = {
                    'bbox' : ped.bbox,
                    'looking' : "Outside",
                    'loc' : outside_coord_avg
                }
                cnt_out_ped += 1
                continue
            if ped.gto is None:
                continue
            dst[ped.token] = {
                'bbox' : ped.bbox,
                'looking' : "Object",
                'gto_bbox' : ped.gto.bbox,
                'gto_center' : ped.gto.center,
                'gto_category' : ped.gto.category
            }
            cnt_obj_ped += 1
        if dst != {}:
            export_json(dst, output_dir+'/'+frame.token+'.json')
    print(f"????????????????????????????????????{cnt_out_ped}???")
    print(f"????????????????????????????????????????????????????????????{cnt_obj_ped}???")

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
    duplicated_double = 0
    duplicated_triple = 0
    for ped_record in tqdm(glob(args.ped_ann + '/*.json'), "running record..."):
        frame_id = osp.splitext(osp.basename(ped_record))[0]
        frame = Frame(frame_id)

        # not use?????????????????????
        not_use_ped = []
        if frame_id in not_use_frame.keys():
            not_use_ped = not_use_frame[frame_id]

        # ??????????????????
        ped_annotations: list = read_json(ped_record)
        ped_annotation: dict
        for ped_token, ped_annotation in ped_annotations.items():
            if ped_token in not_use_ped:
                continue
            ped = Pedestrian(ped_token, frame_id, ped_annotation['bbox'])
            ped.set_gaze_point(ped_annotation['look'], ped_annotation['difficult'], ped_annotation['eyecontact'])
            frame.add_ped(ped)
        # ???????????????????????????
        dataset_objects: dict = read_json(args.obj_dataset + '/' + frame_id + '.json')
        object_token: str
        dataset_object: dict
        for object_token, dataset_object in dataset_objects.items():
            obj = GazeTargetObject(object_token, dataset_object['bbox'], dataset_object['category'])
            frame.add_obj(obj)
            double, triple = check_looking(frame.peds, obj)  # ??????????????????????????????????????????????????????????????????
            duplicated_double += double
            duplicated_triple += triple

        # frame?????????????????????
        frame.stat()
        frames[frame_id] = frame

    os.makedirs(args.output, exist_ok=True)
    export_ped_obj_set(frames, args.output)

    # ?????????????????????
    cnt_peds = []
    cnt_objs = []
    cnt_looking_eyecontact = []
    cnt_looking_difficult = []
    cnt_looking_inside_frame = []
    cnt_looking_inside_obj = []
    cnt_gaze_points = 0
    cnt_gaze_looking_duplicated = 0
    looking_multi_obj = []
    lmo = {}
    frame: Frame
    for frame in tqdm(frames.values(), "statistics..."):
        cnt_peds.append(len(frame.peds))
        cnt_objs.append(len(frame.objects))
        cnt_looking_eyecontact.append(frame.looking_eyecontact)
        cnt_looking_difficult.append(frame.looking_difficult)
        cnt_looking_inside_frame.append(frame.looking_inside_frame)
        cnt_looking_inside_obj.append(frame.looking_inside_obj)
        if len(frame.looking_multi_obj) > 0:
            looking_multi_obj.extend([token+'@'+frame.token for token in frame.looking_multi_obj])
            lmo[frame.token] = frame.looking_multi_obj
        ped: Pedestrian
        for ped in frame.peds:
            cnt_gaze_points += len(ped.gaze)
            gaze: GazePoint
            for gaze in ped.gaze:
                if len(gaze.gto) >= 2:
                    cnt_gaze_looking_duplicated += 1
    print(f"??????????????? -> {sum(cnt_peds)}")
    if len(cnt_peds) > 1:
        print(f"???????????????????????????????????? -> {statistics.mean(cnt_peds):.02f} \u00B1 {statistics.stdev(cnt_peds):.02f}")
    print(f"???????????????????????? -> {sum(cnt_objs)}")
    if len(cnt_objs) > 1:
        print(f"????????????????????????????????????????????? -> {statistics.mean(cnt_objs):.02f} \u00B1 {statistics.stdev(cnt_objs):.02f}")
    print(f"?????????????????????????????? -> {cnt_gaze_points}({(cnt_gaze_points*100/(sum(cnt_peds)*3)):.02f}%)")
    print(f"??????????????????????????? -> {sum(cnt_looking_eyecontact)}({(sum(cnt_looking_eyecontact)*100/cnt_gaze_points):.02f}%)")
    print(f"difficult?????? -> {sum(cnt_looking_difficult)}({(sum(cnt_looking_difficult)*100/cnt_gaze_points):.02f}%)")
    print(f"???????????????????????????????????? -> {sum(cnt_looking_inside_frame)}({(sum(cnt_looking_inside_frame)*100/cnt_gaze_points):.02f}%)")
    print(f"?????????????????????????????????????????? -> {sum(cnt_looking_inside_obj)}({(sum(cnt_looking_inside_obj)*100/cnt_gaze_points):.02f}%)")
    print(f"2??????????????????????????????????????????????????????(3??????????????????) -> {duplicated_double}??????????????????({(duplicated_double*100/sum(cnt_objs)):.02f}%)")
    print(f"3?????????????????????????????????????????????????????? -> {duplicated_triple}??????????????????({(duplicated_triple*100/sum(cnt_objs)):.02f}%)")
    print(f"1??????????????????????????????????????????????????????????????? -> {cnt_gaze_looking_duplicated}???({(cnt_gaze_looking_duplicated*100/cnt_gaze_points):.02f}%)")
    print(f"2?????????????????????????????????????????????????????????????????????????????? -> {len(looking_multi_obj)}???")
    export_json(lmo, args.output+'/looking_multi_obj.json')

if __name__ == "__main__":
    main()