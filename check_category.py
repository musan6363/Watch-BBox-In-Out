import json
from glob import glob

def read_json(json_path: str) -> list:
    with open(json_path, 'r') as f:
        _data = json.load(f)
    return _data

category = []
for rec in glob("/home0/murakamih/work/Watch-BBox-In-Out/object-extract/output/Waymo/validation/*.json"):
    data = read_json(rec)
    for d in data.values():
        category.append(d['category'])

print(set(category))