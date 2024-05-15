# TextBite

Tool for automatic topic segmentation of text pages in the Czech language

## Authors

Martin Kostelník (xkoste12@stud.fit.vutbr.cz)

Karel Beneš (ibenes@fit.vut.cz)

## Running

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

Should you only run YOLO/GNN as standalone models, the scripts are available in the `scripts` folder.

## Models

The models are available [online](https://nextcloud.fit.vutbr.cz/s/6jNgze6fLYXQBgq).
