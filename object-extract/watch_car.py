# 車が注視対象の歩行者を描画する
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

def watch_self(bbox: list, p: list) -> int:
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

def render(dataset_name, version_name, record_name, ped, bbox, ps, gto_bbox):
    img_path = glob(f"{img_root}/{dataset_name}/{version_name}/img/{record_name}.*")[0]

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    px = [p[0] for p in ps]
    py = [p[1] for p in ps]

    ax.scatter(px, py, c='red', s=50)
    r1 = patches.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], ec='yellow', fill=False)
    ax.add_patch(r1)
    r2 = patches.Rectangle(xy=(gto_bbox[0], gto_bbox[1]), width=gto_bbox[2]-gto_bbox[0], height=gto_bbox[3]-gto_bbox[1], ec='yellow', fill=False)
    ax.add_patch(r2)
    im = Image.open(img_path)
    ax.imshow(im)

    savepath = f"../output0121_2/{dataset_name}/{version_name}/{record_name}_{ped}.png"
    savedir = os.path.dirname(savepath)
    os.makedirs(savedir, exist_ok='True')
    plt.savefig(savepath)
    plt.close()

ann_path = "/work/murakamih/annotation/v230105"
lp_ann_path = "../annotation"
img_root = "/work/murakamih/pedestrian"  # /work/murakamih/pedestrian/nuimages_ped_1017/v1.0-train/img/

for dataset in glob(ann_path + '/*'):
    dataset_name = dataset.split('/')[-1]
    for version in glob(dataset + '/*'):
        version_name = version.split('/')[-1]
        for records in tqdm(glob(version + '/*.json')):
            record_name = records.split('/')[-1][:-5]
            if record_name == "looking_multi_obj":
                continue
            anns = read_json(records)
            for ped, ann in anns.items():
                bbox = ann['bbox']
                if ann['looking'] == "Object" and ann["gto_category"] == "vehicle.car":
                    tmp_rec = read_json(f"{lp_ann_path}/{dataset_name}/{version_name}/{record_name}.json")
                    for p, a in tmp_rec.items():
                        if ped == p:
                            lp = a['look']
                    render(dataset_name, version_name, record_name, ped, bbox, lp, ann['gto_bbox'])