import cv2, sys, os
from pathlib import Path

names = ["Car","Van","Truck","Pedestrian","Person_sitting","Cyclist","Tram","Misc"]

img_path = Path(sys.argv[1])
lbl_path = Path(sys.argv[2])

img = cv2.imread(str(img_path))
h, w = img.shape[:2]
for line in lbl_path.read_text().splitlines():
    if not line.strip(): continue
    cid, x, y, bw, bh = map(float, line.split())
    cx, cy = x*w, y*h
    ww, hh = bw*w, bh*h
    l = int(cx - ww/2); t = int(cy - hh/2)
    r = int(cx + ww/2); b = int(cy + hh/2)
    cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
    cls = names[int(cid)]
    cv2.putText(img, cls, (l, max(0,t-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
cv2.imwrite("vis.jpg", img)
print("Saved vis.jpg")