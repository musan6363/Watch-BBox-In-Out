from tqdm import tqdm
import json
import os
import sys

VERSION = sys.argv[1]
SAVE_DIR = 'output/Waymo/' + VERSION
VALID_CAMERE_DIRECTIONS = ['CAM_FRONT', 'CAM_BACK']

class ObjectAnn:
    '''
    nuImagesのAnnotation:object_annに相当
    データセット内でアノテーションされた各オブジェクトに相当
    '''
    def __init__(self, token, category_token, bbox) -> None:
        self.token = token
        self.category = self._get_category_name(category_token)
        self.bbox = bbox

    def _get_category_name(self, catogory_token) -> str:
        # category = nuim.get('category', catogory_token)
        # return category['name']
        pass

    def is_ped(self) -> bool:
        # if self.category[:16] == "human.pedestrian":
        #     ped_height = self.bbox[3] - self.bbox[1]
        #     if ped_height >= 200:
        #         return True
        return False

class SampleData:
    '''
    nuImagesのExtraction:sample_dataに相当
    各画像を表す
    '''
    def __init__(self, token: str) -> None:
        self.token = token
        self.obj = []

    def append_obj(self, obj: ObjectAnn) -> None:
        self.obj.append(obj)

    def is_valid(self) -> bool:
        return self._is_valid_camera() and self._is_include_ped()

    def _is_include_ped(self) -> bool:
        _is_ped_found = False
        for obj in self.obj:
            if obj.is_ped():
                _is_ped_found = True
                break
        return _is_ped_found

    def _is_valid_camera(self) -> bool:
        # _sample_data = nuim.get('sample_data', self.token)
        # _calibrated_sensor = nuim.get('calibrated_sensor', _sample_data['calibrated_sensor_token'])
        # _sensor = nuim.get('sensor', _calibrated_sensor['sensor_token'])
        # _camera_direction = _sensor['channel']
        # return _camera_direction in VALID_CAMERE_DIRECTIONS
        pass

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
    sample_data_list = {}
    for obj in tqdm(nuim.object_ann, "Scanning Objects..."):
        objann = ObjectAnn(obj['token'], obj['category_token'], obj['bbox'])
        sample_data_token = obj['sample_data_token']
        if not sample_data_token in sample_data_list.keys():
            sd = SampleData(sample_data_token)
            sample_data_list[sample_data_token] = sd
        else:
            sd = sample_data_list[sample_data_token]
        sd.append_obj(objann)

    os.makedirs(SAVE_DIR, exist_ok=True)
    sd: SampleData
    for sd in tqdm(sample_data_list.values(), "Scanning records that include pedestrians..."):
        if sd.is_valid():
            sd.export(SAVE_DIR + '/' + sd.token + '.json')


if __name__ == "__main__":
    main()