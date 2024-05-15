from __future__ import annotations

"""Page Geometry definition

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


from typing import Optional, List, Tuple, Dict
from collections import namedtuple
from math import sqrt
from functools import cached_property

import numpy as np
import torch

from shapely.ops import nearest_points
from shapely.geometry import Polygon

from pero_ocr.document_ocr.layout import PageLayout, TextLine

from textbite.utils import hash_strings


Point = namedtuple("Point", "x y")
AABB = namedtuple("AABB", "xmin ymin xmax ymax")


def dist_l2(p1: Point, p2: Point) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return sqrt(dx*dx + dy*dy)


def bbox_dist_y(bbox1: AABB, bbox2: AABB) -> float:
    bbox1_center_y = bbox_center(bbox1).y
    bbox2_center_y = bbox_center(bbox2).y

    bbox1_half_height = bbox1.ymax - bbox1_center_y
    bbox2_half_height = bbox2.ymax - bbox2_center_y

    return max(0.0, abs(bbox1_center_y - bbox2_center_y) - bbox1_half_height - bbox2_half_height)


def polygon_to_bbox(polygon: np.ndarray) -> AABB:
    mins = np.min(polygon, axis=0)
    maxs = np.max(polygon, axis=0)

    # (minx, miny, maxx, maxy)
    return AABB(int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]))


def enclosing_bbox(bboxes: List[AABB]) -> AABB:
    xmins = [bbox.xmin for bbox in bboxes]
    xmaxs = [bbox.xmax for bbox in bboxes]
    ymins = [bbox.ymin for bbox in bboxes]
    ymaxs = [bbox.ymax for bbox in bboxes]

    bbox = AABB(min(xmins), max(xmaxs), min(ymins), max(ymaxs))
    return bbox


def bbox_center(bbox: AABB) -> Point:
    x = (bbox.xmin + ((bbox.xmax - bbox.xmin) / 2))
    y = (bbox.ymin + ((bbox.ymax - bbox.ymin) / 2))

    return Point(x, y)


def bbox_area(bbox: AABB) -> float:
    return float((bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin))


def bbox_intersection(lhs: AABB, rhs: AABB) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    dy = min(lhs.ymax, rhs.ymax) - max(lhs.ymin, rhs.ymin)

    return dx * dy if dx >= 0.0 and dy >= 0.0 else 0.0


def bbox_intersection_over_area(lhs: AABB, rhs: AABB) -> float:
    intersection = bbox_intersection(lhs, rhs)
    area = bbox_area(lhs)

    assert intersection <= area
    return intersection / area


def bbox_intersection_x(lhs: AABB, rhs: AABB) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    return max(dx, 0.0)


def best_intersecting_bbox(target_bbox: AABB, candidate_bboxes: List[AABB]):
    best_region = None
    best_intersection = 0.0
    for i, bbox in enumerate(candidate_bboxes):
        intersection = bbox_intersection(target_bbox, bbox)
        if intersection > best_intersection:
            best_intersection = intersection
            best_region = i

    return best_region


def bbox_to_yolo(bbox: AABB, page_width, page_height) -> Tuple[float, float, float, float]:
    dx, dy = bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin
    x = (bbox.xmin + (dx / 2.0)) / page_width
    y = (bbox.ymin + (dy / 2.0)) / page_height
    width = dx / page_width
    height = dy / page_height

    return x, y, width, height


class Ray:
    def __init__(self, origin: Point, direction: Point):
        self.origin = origin
        self.direction = direction

        length = sqrt(self.direction.x*self.direction.x + self.direction.y*self.direction.y)
        x = self.direction.x / length
        y = self.direction.y / length
        self.direction = Point(x, y)

    def intersects_bbox(self, bbox: AABB) -> Optional[float]:
        if self.direction.x == 0 and (self.origin.x < bbox.xmin or self.origin.x > bbox.xmax):
            return None

        if self.direction.y == 0 and (self.origin.y < bbox.ymin or self.origin.y > bbox.ymax):
            return None

        tmin = -float('inf')
        tmax = float('inf')

        if self.direction.x != 0:
            tx1 = (bbox.xmin - self.origin.x) / self.direction.x
            tx2 = (bbox.xmax - self.origin.x) / self.direction.x

            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        if self.direction.y != 0:
            ty1 = (bbox.ymin - self.origin.y) / self.direction.y
            ty2 = (bbox.ymax - self.origin.y) / self.direction.y

            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmin <= tmax and tmax >= 0:
            return max(tmin, 0)
        else:
            return None


def find_visible_entities(rays: List[Ray], entities: List[GeometryEntity]) -> List[GeometryEntity]:
    visible_entities = []

    for ray in rays:
        best_dist = float("inf")
        closest_entity = None
        for entity in entities:
            dist = ray.intersects_bbox(entity.bbox)
            if dist:
                if dist < best_dist:
                    best_dist = dist
                    closest_entity = entity

        if closest_entity and closest_entity not in visible_entities:
            visible_entities.append(closest_entity)

    return visible_entities


class GeometryEntity:
    def __init__(self, page_geometry: Optional[PageGeometry]=None):
        self.page_geometry = page_geometry # Reference to the geometry of the entire page

        self.parent: Optional[GeometryEntity] = None
        self.child: Optional[GeometryEntity] = None
        self.neighbourhood: Optional[List[GeometryEntity]] = None
        self.visible_entities: Optional[List[GeometryEntity]] = None

    @property
    def bbox(self) -> AABB:
        ...
        
    @property
    def width(self) -> float:
        return self.bbox.xmax - self.bbox.xmin

    @property
    def height(self) -> float:
        return self.bbox.ymax - self.bbox.ymin

    @property
    def center(self) -> Point:
        return bbox_center(self.bbox)

    @property
    def bbox_area(self):
        return bbox_area(self.bbox)
    
    @property
    def number_of_predecessors(self) -> int:
        return sum([1 for _ in self.parent_iterator()])

    @property
    def number_of_successors(self) -> int:
        return sum([1 for _ in self.children_iterator()])
    
    @property
    def vertical_neighbours(self, neighbourhood_size: int) -> List[LineGeometry]:
        neighbourhood = [self]
        parent_ptr = self.parent
        child_ptr = self.child

        for _ in range(neighbourhood_size):
            if parent_ptr:
                neighbourhood.append(parent_ptr)
                parent_ptr = parent_ptr.parent

            if child_ptr:
                neighbourhood.append(child_ptr)
                child_ptr = child_ptr.child

        return neighbourhood

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

    def lineage_iterator(self):
        for parent in self.parent_iterator():
            yield parent
        for child in self.children_iterator():
            yield child            
    
    def set_parent(self, entities: List[GeometryEntity], threshold: float=0.0) -> None:
        parent_candidates = [entity for entity in entities if self is not entity]
        # Filter entities below me
        parent_candidates = [entity for entity in parent_candidates if entity.center.y < self.center.y]
        # Filter entities that have no horizontal overlap with me
        parent_candidates = [entity for entity in parent_candidates if bbox_intersection_x(self.bbox, entity.bbox) > threshold]
        if parent_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the highest Y values
            self.parent = max(parent_candidates, key=lambda x: x.center.y)

    def set_child(self, entities: List[GeometryEntity], threshold: int=0.0) -> None:
        child_candidates = [entity for entity in entities if self is not entity]
        # Filter entities above me
        child_candidates = [entity for entity in child_candidates if entity.center.y > self.center.y]
        # Filter entities that have no horizontal overlap with me
        child_candidates = [entity for entity in child_candidates if bbox_intersection_x(self.bbox, entity.bbox) > threshold]
        if child_candidates:
            # Take the candidate, which is closest to me in Y axis <==> The one with the lowest Y values
            self.child = min(child_candidates, key=lambda x: x.center.y)

    def set_visibility(self, entities: List[GeometryEntity]) -> None:
        ...


class RegionGeometry(GeometryEntity):
    def __init__(self, bbox: AABB, page_geometry: Optional[PageGeometry]):
        super().__init__(page_geometry)
        self._bbox = bbox

    @property
    def bbox(self) -> AABB:
        assert self._bbox.xmax > self._bbox.xmin and self._bbox.ymax > self._bbox.ymin
        return self._bbox
    
    def set_visibility(self, entities: List[GeometryEntity]) -> None:
        assert self.page_geometry is not None

        self.visible_entities = []
        other_entities = [entity for entity in entities if self is not entity]

        if self.parent is not None:
            self.visible_entities.append(self.parent)

        if self.child is not None:
            self.visible_entities.append(self.child)

        # Create horizontal rays
        horizontal_rays = []
        horizontal_rays.append(Ray(Point(self.bbox.xmax, self.center.y), Point(1, 0.5)))
        horizontal_rays.append(Ray(Point(self.bbox.xmax, self.center.y), Point(1, 0)))
        horizontal_rays.append(Ray(Point(self.bbox.xmax, self.center.y), Point(1, -0.5)))

        horizontal_visible_entities = find_visible_entities(horizontal_rays, other_entities)
        self.visible_entities.extend(horizontal_visible_entities)
        for ve in horizontal_visible_entities:
            for relative in ve.lineage_iterator():
                if relative not in self.visible_entities:
                    self.visible_entities.append(relative)

        self.visible_entities = list(set(self.visible_entities))


class LineGeometry(GeometryEntity):
    def __init__(self, text_line: TextLine, page_geometry: Optional[PageGeometry]):
        super().__init__(page_geometry)

        self.text_line: TextLine = text_line
        self.polygon = text_line.polygon

    @cached_property
    def bbox(self) -> AABB:
        _bbox = polygon_to_bbox(self.text_line.polygon)
        assert _bbox.xmax > _bbox.xmin and _bbox.ymax > _bbox.ymin
        return _bbox
    
    def set_neighbourhood(self, lines: List[LineGeometry], max_neighbours: int, cached_distances: Dict[str, float]={}) -> None:
        distances = []
        for other_line in lines:
            if self is other_line:
                continue
            hash = hash_strings(self.text_line.id, other_line.text_line.id)
            try:
                distance = cached_distances[hash]
            except KeyError:
                nps = nearest_points(Polygon(self.text_line.polygon), Polygon(other_line.text_line.polygon))
                p1 = Point(nps[0].x, nps[0].y)
                p2 = Point(nps[1].x, nps[1].y)
                distance = dist_l2(p1, p2)
                cached_distances[hash] = distance

            distances.append((other_line, distance))

        self.neighbourhood = []
        mx = max_neighbours if len(distances) > max_neighbours else len(distances)
        while len(self.neighbourhood) != mx:
            m = min(distances, key=lambda x: x[1])
            self.neighbourhood.append(m[0])
            distances.remove(m)

    def set_visibility(self, entities: List[GeometryEntity]) -> None:
        assert self.page_geometry is not None

        self.visible_entities = []
        N_HORIZONTAL_RAYS = 100
        y_span = self.page_geometry.page_height * 0.1
        y_step = y_span / N_HORIZONTAL_RAYS
        other_entities = [entity for entity in entities if self is not entity]

        # Create vertical rays
        vertical_rays = []
        epsilon = self.width * 0.05
        vertical_rays.append(Ray(Point(self.bbox.xmin + epsilon, self.center.y), Point(0, 1)))
        vertical_rays.append(Ray(Point(self.center.x, self.center.y), Point(0, 1)))
        vertical_rays.append(Ray(Point(self.bbox.xmax - epsilon, self.center.y), Point(0, 1)))
        vertical_rays.append(Ray(Point(self.bbox.xmin + epsilon, self.center.y), Point(0, -1)))
        vertical_rays.append(Ray(Point(self.center.x, self.center.y), Point(0, -1)))
        vertical_rays.append(Ray(Point(self.bbox.xmax - epsilon, self.center.y), Point(0, -1)))

        # Add all lines intersected by vertical rays to visible_entities
        vertical_visible_entities = find_visible_entities(vertical_rays, other_entities)
        self.visible_entities.extend(vertical_visible_entities)

        # Create horizontal rays
        horizontal_rays = []
        mid_point_y = self.bbox.ymin + ((self.bbox.ymax - self.bbox.ymin) / 2.0)

        # Horizontal rays (current)
        y_start = 0.0 - (y_span / 2.0)
        y = y_start
        for _ in range(N_HORIZONTAL_RAYS):
            horizontal_rays.append(Ray(Point(self.bbox.xmax, mid_point_y), Point(70, y)))
            y += y_step

        horizontal_visible_entities = find_visible_entities(horizontal_rays, other_entities)
        self.visible_entities.extend(horizontal_visible_entities)
        for entity in horizontal_visible_entities:
            if entity.child and entity.child not in self.visible_entities and self is not entity.child:
                self.visible_entities.append(entity.child)
            if entity.parent and entity.parent not in self.visible_entities and self is not entity.parent:
                self.visible_entities.append(entity.parent)

        self.visible_entities = list(set(self.visible_entities))
    
    def __str__(self) -> str:
        output_str = "LineGeometry object"
        output_str += f"\nCenter = {self.center}"
        output_str += f"\nWidth, height = {self.get_width()}, {self.get_height()}"
        output_str += f"\nBBox = {self.bbox}"
        output_str += f"\nNumber of parents = {self.get_number_of_predecessors()}"
        output_str += f"\nNumber of children = {self.get_number_of_successors()}"

        return output_str


class PageGeometry:
    def __init__(
            self,
            regions: List[AABB]=[],
            path: Optional[str]=None,
            pagexml: Optional[PageLayout]=None,
        ):
        self.pagexml: PageLayout = pagexml
        if path:
            self.pagexml = PageLayout(file=path)

        self.lines = []
        self.regions = [RegionGeometry(region, self) for region in regions]

        if self.pagexml is not None:
            self.lines: List[LineGeometry] = [LineGeometry(line, self) for line in self.pagexml.lines_iterator() if line.transcription and line.transcription.strip()]
            self.lines_by_id = {line.text_line.id: line for line in self.lines}

            h, w = self.pagexml.page_size
            self.page_width = w
            self.page_height = h
        
        for line in self.lines:
            line.set_parent(self.lines)
            line.set_child(self.lines)

        for region in self.regions:
            region.set_parent(self.regions, threshold=10)
            region.set_child(self.regions, threshold=10)

    @property
    def page_area(self):
        return self.page_width * self.page_height

    @property
    def avg_line_width(self) -> float:
        if not self.lines or len(self.lines) == 0:
            raise ValueError("No lines exist in this PageGeometry.")
        return sum(line.get_width() for line in self.lines) / len(self.lines)

    @property
    def avg_line_height(self) -> float:
        if not self.lines or len(self.lines) == 0:
            raise ValueError("No lines exist in this PageGeometry.")
        return sum(line.get_height() for line in self.lines) / len(self.lines)
    
    @property
    def line_heads(self) -> List[LineGeometry]:
        return [line_geometry for line_geometry in self.lines if line_geometry.parent is None]
    
    @property
    def avg_line_distance_y(self) -> float:
        processed_pairs = []
        distance_sum = 0.0

        for line_geometry in self.lines:
            if line_geometry.parent is not None:
                if set([line_geometry, line_geometry.parent]) not in processed_pairs:
                    distance_sum += bbox_dist_y(line_geometry.bbox, line_geometry.parent.bbox)
                    processed_pairs.append(set([line_geometry, line_geometry.parent]))

            if line_geometry.child is not None:
                if set([line_geometry, line_geometry.child]) not in processed_pairs:
                    distance_sum += bbox_dist_y(line_geometry.bbox, line_geometry.child.bbox)
                    processed_pairs.append(set([line_geometry, line_geometry.child]))

        return distance_sum / len(processed_pairs)


    def set_line_neighbourhoods(self, max_neighbours: int=10) -> None:
        cached_distances = {}
        for line in self.lines:
            other_lines = [l for l in self.lines if not (l is line)]
            line.set_neighbourhood(other_lines, max_neighbours, cached_distances)

    def set_line_visibility(self) -> None:
        for line in self.lines:
            other_lines = [l for l in self.lines if l is not line]
            line.set_visibility(other_lines)

    def set_region_visibility(self) -> None:
        for region in self.regions:
            other_regions = [r for r in self.regions if r is not region]
            region.set_visibility(other_regions)

    def split_geometry(self) -> None:
        for line_geometry in self.lines:
            # If line is a parent to multiple lines, sever all the connections
            children = [potentional_child for potentional_child in self.lines if potentional_child.parent is line_geometry]
            if len(children) >= 2:
                line_geometry.child = None
                for child in children:
                    child.parent = None

            # If line a child of multiple lines, sever all
            parents = [potentional_parent for potentional_parent in self.lines if potentional_parent.child is line_geometry]
            if len(parents) >= 2:
                line_geometry.parent = None
                for parent in parents:
                    parent.child = None

            # If line A has a child B and there exist a different line C, whose parent is A, sever all
            if line_geometry.child is not None:
                for potentional_child in self.lines:
                    if potentional_child is line_geometry.child:
                        continue

                    if potentional_child.parent is line_geometry:
                        line_geometry.child = None
                        potentional_child.parent = None

            # If line A has parent B and there exists a different line C, whose child is A, sever all
            if line_geometry.parent is not None:
                for potentional_parent in self.lines:
                    if potentional_parent is line_geometry.parent:
                        continue

                    if potentional_parent.child is line_geometry:
                        line_geometry.parent = None
                        potentional_parent.child = None

        for line_geometry in self.lines:
            if line_geometry.parent is not None:
                if line_geometry.parent.child is None:
                    line_geometry.parent = None

            if line_geometry.child is not None:
                if line_geometry.child.parent is None:
                    line_geometry.child = None

        for line_geometry in self.lines:
            if line_geometry.parent is not None:
                if line_geometry.parent.child is not line_geometry:
                    line_geometry.parent.child = None
                    line_geometry.parent = None

            if line_geometry.child is not None:
                if line_geometry.child.parent is not line_geometry:
                    line_geometry.child.parent = None
                    line_geometry.child = None


    def __str__(self) -> str:
        output_str = f"PageGeometry object"
        output_str += f"\n{len(self.lines)} lines."
        output_str += f"\n{len(self.regions)} regions."
        output_str += f"\nPage size is {self.page_width}x{self.page_height}"

        return output_str


if __name__ == "__main__":
    EXAMPLE_PATH = r"/home/martin/textbite/data/segmentation/xmls/train/cesky-zapad-(2)-08.xml"
    pagexml = PageLayout(file=EXAMPLE_PATH)

    geometry1 = PageGeometry(pagexml=pagexml)

    geometry1.set_line_neighbourhoods()
    geometry1.set_line_visibility()
    geometry1.set_region_visibility()

    for line in geometry1.lines:
        print(len(line.visible_entities))

    for region in geometry1.regions:
        print(len(region.visible_entities))
