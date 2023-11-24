from collections import namedtuple
from typing import Tuple, List
import os
from time import perf_counter

import numpy as np
import cv2

from pero_ocr.document_ocr.layout import PageLayout


PATH = r"/home/martin/textbite/tmp/cesky-zapad-(1)-5.xml"


class Line:
    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

    def __init__(self, text_line):
        self.text_line = text_line
        self.parent = None
        self.child = None
        
        self.set_geometry()


    def set_geometry(self) -> None:
        mins = np.min(self.text_line.polygon, axis=0)
        maxs = np.max(self.text_line.polygon, axis=0)

        # (minx, miny, maxx, maxy)
        self.bbox = Line.Rectangle(mins[0], mins[1], maxs[0], maxs[1])

        x = self.bbox.xmin + ((self.bbox.xmax - self.bbox.xmin) / 2.0)
        y = self.bbox.ymin + ((self.bbox.ymax - self.bbox.ymin) / 2.0)

        self.center = (x, y)

    def set_parent(self, lines):
        x, y = self.center[0], self.center[1]
        parent_candidates = [line for line in lines if line.center[1] < y]
        parent_candidates = [line for line in parent_candidates if self.x_candidate_predicate(line)]
        if parent_candidates:
            asdf = sorted(parent_candidates, reverse=True, key=lambda x: x.center[1])
            self.parent = sorted(parent_candidates, reverse=True, key=lambda x: x.center[1])[0]

    def set_child(self, lines):
        x, y = self.center[0], self.center[1]
        child_candidates = [line for line in lines if line.center[1] > y]
        child_candidates = [line for line in child_candidates if self.x_candidate_predicate(line)]
        if child_candidates:
            self.child = sorted(child_candidates, key=lambda x: x.center[1])[0]

    def x_candidate_predicate(self, ref) -> bool:
        return    (self.bbox.xmin >= ref.bbox.xmin and self.bbox.xmin <= ref.bbox.xmax) \
               or (self.bbox.xmax >= ref.bbox.xmin and self.bbox.xmax <= ref.bbox.xmax) \
               or (ref.bbox.xmin >= self.bbox.xmin and ref.bbox.xmin <= self.bbox.xmax) \
               or (ref.bbox.xmax >= self.bbox.xmin and ref.bbox.xmax <= self.bbox.xmax)

    def children_iterator(self):
        ptr = self.child
        while ptr:
            yield ptr
            ptr = ptr.child

    def parent_iterator(self):
        ptr = self.parent
        while ptr:
            yield ptr
            ptr = ptr.parent


def main():
    img = cv2.imread(PATH.replace(".xml", ".jpg"))

    start = perf_counter()
    pagexml = PageLayout(file=PATH)

    lines = [Line(line) for line in pagexml.lines_iterator()]

    for line in lines:
        line.set_parent(lines)
        line.set_child(lines)

    end = perf_counter()
    print(end - start)

    for line in lines:
        if line.parent:
            start = (int(line.center[0]) + 20, int(line.center[1]))
            end = (int(line.parent.center[0]) + 20, int(line.parent.center[1]))
            cv2.line(img, start, end, (0, 0, 255), 7)

        if line.child:
            start = (int(line.center[0]), int(line.center[1]))
            end = (int(line.child.center[0]), int(line.child.center[1]))
            cv2.line(img, start, end, (0, 255, 0), 7)


    pth = os.path.join("/home/martin/textbite/geometry", "cesky-zapad-(1)-5.jpg")
    cv2.imwrite(pth, img)


if __name__ == "__main__":
    main()
