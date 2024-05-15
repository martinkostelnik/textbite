# TextBite

Tool for automatic topic segmentation of text pages in the Czech language

## Authors

Martin Kostelník (xkoste12@stud.fit.vutbr.cz)

Karel Beneš (ibenes@fit.vut.cz)

## Running

Running requires PERO-OCR installed

To perform YOLOv8+GNN inference, run:
```
textbite \
    --xml xmls/ \
    --img imgs/ \
    --yolo yolo.pt \
    --gnn gnn.pth \
    --normalizer norm.pkl \
    --save out/ \
    [--logging-level INFO] \
    [--json] \
    [--alto altos/]
 ```

Should you only run YOLO/GNN as standalone models, bash scripts are available in the `scripts` folder. Note that paths in those scripts have to be changed to suit your needs

## Models

The models are available [online](https://nextcloud.fit.vutbr.cz/s/6jNgze6fLYXQBgq).
