# 自分自身のBBox内にアノテーション点がある歩行者の抜粋
import os
from glob import glob
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

def render(dataset_name, version_name, record_name, ped, bbox, ps):
    img_path = glob(f"{img_root}/{dataset_name}/{version_name}/img/{record_name}.*")[0]

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    px = [p[0] for p in ps]
    py = [p[1] for p in ps]

    ax.scatter(px, py, c='red', s=50)
    r = patches.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], ec='yellow', fill=False)
    ax.add_patch(r)
    im = Image.open(img_path)
    ax.imshow(im)

    savepath = f"../output0121/{dataset_name}/{version_name}/{record_name}_{ped}.png"
    savedir = os.path.dirname(savepath)
    os.makedirs(savedir, exist_ok='True')
    plt.savefig(savepath)
    plt.close()

ann_path = "../annotation"
img_root = "/work/murakamih/pedestrian"  # /work/murakamih/pedestrian/nuimages_ped_1017/v1.0-train/img/

for dataset in glob(ann_path + '/*'):
    dataset_name = dataset.split('/')[-1]
    for version in glob(dataset + '/*'):
        version_name = version.split('/')[-1]
        for records in glob(version + '/*.json'):
            record_name = records.split('/')[-1][:-5]
            anns = read_json(records)
            for ped, ann in anns.items():
                bbox = ann['bbox']
                cnt_watch_self = 0
                for p in ann['look']:
                    cnt_watch_self += watch_self(bbox, p)
                if cnt_watch_self == 3:
                    print(records, ped)
                    render(dataset_name, version_name, record_name, ped, bbox, ann['look'])
