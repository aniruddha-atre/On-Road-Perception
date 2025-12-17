import os
from pathlib import Path
import numpy as np
from PIL import Image

# Cityscapes labelID → trainID mapping
# 255 means ignore
LABELID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

ROOT = Path(r"D:/Projects/On_Road_Perception/data/cityscapes")

labelids_root = ROOT / "gtFine/val"
output_root   = ROOT / "gtFine_trainIds/val"
output_root.mkdir(parents=True, exist_ok=True)

for city_folder in labelids_root.iterdir():
    if not city_folder.is_dir():
        continue

    out_city = output_root / city_folder.name
    out_city.mkdir(exist_ok=True)

    for f in city_folder.glob("*_labelIds.png"):
        arr = np.array(Image.open(f))

        trainId = np.zeros_like(arr)
        for k, v in LABELID_TO_TRAINID.items():
            trainId[arr == k] = v

        out_path = out_city / f.name.replace("labelIds", "trainIds")
        Image.fromarray(trainId.astype(np.uint8)).save(out_path)

print("DONE — val trainIds generated.")
