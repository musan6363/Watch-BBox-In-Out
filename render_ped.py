# 自分自身のBBox内にアノテーション点がある歩行者の抜粋
import os
from glob import glob
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

def read_json(json_path: str) -> list:
    with open(json_path, 'r') as f:
        _data = json.load(f)
    return _data

def p_in_bbox(bbox: list, p: list) -> int:
    if len(p) == 0:
        return 0
    try:
        if bbox[0] <= p[0] and p[0] <= bbox[2] and bbox[1] <= p[1] and p[1] <= bbox[3]:
            return 1
        else:
            return 0
    except IndexError as e:
        print(e)
        print(bbox, p)

def render(dataset_name, version_name, record_name, ped, bbox, ps, type, obj_name):
    img_path = glob(f"{img_root}/{dataset_name}/{version_name}/img/{record_name}.*")[0]

    if dataset_name != 'waymo-ped':
        fig = plt.figure(figsize=(16, 9))
    else:
        fig = plt.figure(figsize=(30, 7))
    ax = fig.add_subplot(1, 1, 1)

    off_set = 0  # 機能していない

    try:
        px = [p[0]+off_set for p in ps]
        py = [p[1]+off_set for p in ps]
    except IndexError as e:
        # 対応できていないエラー
        print(f"Index Error -> {e}")
        print(ps)
        plt.close()
        return

    ax.scatter(px, py, c='red', s=350, edgecolors='black')
    r = patches.Rectangle(xy=(bbox[0]+off_set, bbox[1]+off_set), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], ec='yellow', fill=False)
    ax.add_patch(r)
    im = Image.open(img_path)
    ax.imshow(im)

    if obj_name != None:
        obj_name = obj_name.replace('.', '_')
        savepath = f"./output0228/{type}/{obj_name}/{dataset_name}/{version_name}/{record_name}_{ped}.png"
    else:
        savepath = f"./output0228/{type}/{dataset_name}/{version_name}/{record_name}_{ped}.png"
    savedir = os.path.dirname(savepath)
    os.makedirs(savedir, exist_ok='True')
    plt.axis('off')
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(savepath, transparent=True)
    plt.close()

ann_path = "/work/murakamih/annotation/v230105"
lp_ann_path = "./annotation"
img_root = "/work/murakamih/pedestrian"  # /work/murakamih/pedestrian/nuimages_ped_1017/v1.0-train/img/

print("Watch Object or Outside")
for dataset in glob(ann_path + '/*'):
    dataset_name = dataset.split('/')[-1]
    # if dataset_name == 'nuscenes_ped' or dataset_name == 'nuimages_ped_1017':
    #     continue
    for version in glob(dataset + '/*'):
        version_name = version.split('/')[-1]
        for records in tqdm(glob(version + '/*.json'), f'{dataset_name}/{version_name}'):
            record_name = records.split('/')[-1][:-5]
            if record_name == "looking_multi_obj":
                continue
            anns = read_json(records)
            for ped, ann in anns.items():
                bbox = ann['bbox']
                tmp_rec = read_json(f"{lp_ann_path}/{dataset_name}/{version_name}/{record_name}.json")
                for p, a in tmp_rec.items():
                    if ped == p:
                        lp = a['look']
                        break
                if len(lp) != 3:
                    continue
                if ann['looking'] == "Object":
                    render(dataset_name, version_name, record_name, ped, bbox, lp, ann['looking'], ann["gto_category"])
                if ann['looking'] == "Outside":
                    render(dataset_name, version_name, record_name, ped, bbox, lp, ann['looking'], None)

print("Eyecontact or Difficult")
for dataset in glob(lp_ann_path + '/*'):
    dataset_name = dataset.split('/')[-1]
    for version in glob(dataset + '/*'):
        version_name = version.split('/')[-1]
        for records in tqdm(glob(version + '/*.json'), f'{dataset_name}/{version_name}'):
            record_name = records.split('/')[-1][:-5]
            anns = read_json(records)
            for ped, ann in anns.items():
                bbox = ann['bbox']
                cnt_ec = 0
                for ec in ann['eyecontact']:
                    cnt_ec += 1 if ec == "true" else 0
                cnt_df = 0
                for df in ann['difficult']:
                    cnt_df += 1 if df == "true" else 0
                if cnt_ec == 3:
                    render(dataset_name, version_name, record_name, ped, bbox, [[0, 0],[0, 0],[0, 0]], "Eyecontact", None)
                if cnt_df == 3:
                    if len(ann['look']) != 3:
                        continue
                    render(dataset_name, version_name, record_name, ped, bbox, ann['look'], "Difficult", None)