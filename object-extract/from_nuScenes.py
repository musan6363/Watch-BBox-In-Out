from tqdm import tqdm
import json
import os
from os import path as osp
import sys
from glob import glob
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import data_classes

VERSION = sys.argv[1]

nusc = NuScenes(dataroot='/work/murakamih/nuscenes', version=VERSION, verbose=True)

SAVE_DIR = 'output/nuScenes/' + VERSION
VALID_CAMERE_DIRECTIONS = ['CAM_FRONT', 'CAM_BACK']

class ObjectAnn:
    '''
    nuScenesのAnnotation:sample_annotationに相当
    データセット内でアノテーションされた各オブジェクトに相当
    '''
    def __init__(self, token, category, bbox) -> None:
        self.token = token
        self.category = category
        self.bbox = bbox

class SampleData:
    '''
    nuScenesのExtraction:sample_dataに相当
    各画像を表す
    '''
    def __init__(self, token: str) -> None:
        self.token = token
        self.obj = []

    def append_obj(self, obj: ObjectAnn) -> None:
        self.obj.append(obj)

    def export(self, export_json_path: str) -> None:
        _dst = {}
        obj: ObjectAnn
        for obj in self.obj:
            _dst[obj.token] = {
                'bbox' : obj.bbox,
                'category' : obj.category
            }
        with open(export_json_path, 'w') as f:
            json.dump(_dst, f)

def get_target_sample() -> list:
    samples = []
    for scene in nusc.scene:
        _cnt_sample = 0
        _sample_token = scene['first_sample_token']
        _last_sample_token = scene['last_sample_token']
        while(True):
            _sample = nusc.get('sample', _sample_token)
            if _cnt_sample % 10 == 0:
                samples.append(_sample_token)
            if _sample_token == _last_sample_token:
                break
            _sample_token = _sample['next']            
            _cnt_sample += 1
    return samples

def load_image_annotations(image_annotation_path) -> dict:
    with open(image_annotation_path, 'r') as f:
        anns = json.load(f)
    obj_anns = {}
    for img_ann in anns:
        obj_anns[img_ann['sample_annotation_token']] = {
            'bbox' : img_ann['bbox_corners'],
            'category' : img_ann['category_name']
        }
    return obj_anns

def main():
    obj_anns = load_image_annotations('/work/murakamih/nuscenes/v1.0-trainval/image_annotations.json')
    gaze_annotated_records = glob("/home0/murakamih/annotation/v221215/json/nuscenes_ped/v1.0-trainval/json/*.json")
    sample_data_list = []
    for gaze_annotated_record in tqdm(gaze_annotated_records):
        sample_data_token = osp.splitext(osp.basename(gaze_annotated_record))[0]
        sd = SampleData(sample_data_token)
        _, boxes, _ = nusc.get_sample_data(sample_data_token)
        if len(boxes) < 1:
            raise ValueError("boxes less than 1")
        box: data_classes.Box
        for box in boxes:
            obj_token = box.token
            obj_ann = obj_anns[obj_token]
            obj = ObjectAnn(obj_token, obj_ann['category'], obj_ann['bbox'])
            sd.append_obj(obj)
        sample_data_list.append(sd)

    os.makedirs(SAVE_DIR, exist_ok=True)
    sd: SampleData
    for sd in tqdm(sample_data_list):
        sd.export(SAVE_DIR + '/' + sd.token + '.json')


if __name__ == "__main__":
    main()