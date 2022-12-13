from tqdm import tqdm
from glob import glob
import json
import os
import sys
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
VERSION = sys.argv[1]
RECORDS_DIR = '/data/dataset/waymo/' + VERSION
SAVE_DIR = 'output/Waymo/' + VERSION
IMG_WIDTH = 1920
IMG_HEIGHT = 1280
VALID_CAMERE_DIRECTIONS = [open_dataset.CameraName.FRONT_LEFT, open_dataset.CameraName.FRONT, open_dataset.CameraName.FRONT_RIGHT]

class ObjectAnn:
    '''
    データセット内でアノテーションされた各オブジェクトに相当
    '''
    def __init__(self, token, category_token: int, bbox: open_dataset.label_pb2.Box, camera_direction: int) -> None:
        self.token = token
        self.category = self._get_category_name(category_token)
        self.bbox = self._get_bbox_loc(bbox, camera_direction)

    def _get_category_name(self, catogory_token: int) -> str:
        return label_pb2.Label.Type.Name(catogory_token)

    def _get_bbox_loc(self, box: open_dataset.label_pb2.Box, camera_direction: int) -> list:
        _bbox_begin_x = box.center_x - 0.5 * box.length
        _bbox_begin_y = box.center_y - 0.5 * box.width
        _bbox_end_x = _bbox_begin_x + box.length
        _bbox_end_y = _bbox_begin_y + box.width
        if camera_direction == open_dataset.CameraName.FRONT:
            _bbox_begin_x += IMG_WIDTH
            _bbox_end_x += IMG_WIDTH
        elif camera_direction == open_dataset.CameraName.FRONT_RIGHT:
            _bbox_begin_x += IMG_WIDTH * 2
            _bbox_end_x += IMG_WIDTH * 2
        return [_bbox_begin_x, _bbox_begin_y, _bbox_end_x, _bbox_end_y]    

class CameraLabels:
    '''
    Waymoの'open_dataset.Frame().camera_labels'に相当
    各画像を表す
    '''
    def __init__(self, frame: open_dataset.Frame, cnt_frame: int) -> None:
        self.token = frame.context.name + "-" + format(cnt_frame, '03d')
        self.obj = []
        self.frame = frame

    def append_obj(self, ann_label: open_dataset.label_pb2.Label, camera_direction: int) -> None:
        obj = ObjectAnn(ann_label.id, ann_label.type, ann_label.box, camera_direction)
        self.obj.append(obj)

    def is_include_ped(self) -> bool:
        _is_ped_found = False
        for cam_obj in self.frame.context.stats.camera_object_counts:
            if cam_obj.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
                _is_ped_found = True
                break
        return _is_ped_found

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

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tfrecord: str
    tfrecords = glob(RECORDS_DIR + '/*.tfrecord')
    for tfrecord in tqdm(tfrecords, "tfrecord : "):
        dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
        cnt_frame = 0
        for data in tqdm(dataset, "dataset : "): 
            if cnt_frame % 50 != 0:
                # 50frame(=5s)ごとに書き出し
                cnt_frame += 1
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            cl = CameraLabels(frame, cnt_frame)
            if not cl.is_include_ped():
                # 本来はここで`cnt_frame += 1`をすべきだが，元の歩行者抜粋プログラムで行っていないため無視
                continue
            camera_labels: open_dataset.dataset_pb2.CameraLabels
            for camera_labels in frame.camera_labels:
                camera_direction = camera_labels.name
                if not camera_direction in VALID_CAMERE_DIRECTIONS:
                    continue
                for ann_label in camera_labels.labels:
                    cl.append_obj(ann_label, camera_direction)
            cl.export(SAVE_DIR + '/' + cl.token + '.json')
            cnt_frame += 1


if __name__ == "__main__":
    main()