from typing import List, Optional

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pero_ocr.document_ocr.layout import PageLayout

from textbite.geometry import AABB, bbox_area, bbox_center


class TextFeaturesProvider:
    def __init__(self):
        pass

    def get_tfidf_features(self, transcriptions: List[str]) -> List[torch.FloatTensor]:
        vectorizer = TfidfVectorizer(max_features=64)

        try:
            tfidf_matrix = vectorizer.fit_transform(transcriptions)
        except ValueError:
            raise RuntimeError("XML Transcriptions corrupted")
        
        tfidf_list = tfidf_matrix.todense(order="C").tolist()

        for item in tfidf_list:
            if len(item) != 64:
                to_add = 64 - len(item)
                item.extend([0.0] * to_add)

        tfidf_list = [torch.tensor(features, dtype=torch.float32) for features in tfidf_list]

        return tfidf_list


class GeometryFeaturesProvider:
    def __init__(self):
        self.pagexml = None
        self.page_geometry = None

    def get_regions_features(self, regions: List[AABB], pagexml: PageLayout) -> List[torch.FloatTensor]:
        self.pagexml = pagexml
        return [self.get_region_features(region) for region in regions]

    def get_region_features(self, region: AABB) -> torch.FloatTensor:
        assert self.pagexml is not None
        page_height, page_width = self.pagexml.page_size

        center = bbox_center(region)
        feature_center_x = center.x
        feature_center_y = center.y

        feature_xmin = float(region.xmin)
        feature_xmax = float(region.xmax)
        feature_ymin = float(region.ymin)
        feature_ymax = float(region.ymax)

        feature_area = bbox_area(region)
        feature_area_relative = feature_area / (page_width * page_height)

        feature_width = feature_xmax - feature_xmin
        feature_width_relative = feature_width / page_width

        feature_height = feature_ymax - feature_ymin
        feature_height_relative = feature_height / page_height

        feature_ratio = feature_width / feature_height

        features = [
            feature_center_x,
            feature_center_y,
            feature_xmin,
            feature_xmax,
            feature_ymin,
            feature_ymax,
            feature_area,
            feature_width,
            feature_height,
            feature_ratio,
            feature_area_relative,
            feature_width_relative,
            feature_height_relative,
        ]

        return torch.tensor(features, dtype=torch.float32)

def get_features(self, return_type: Optional[str]=None) -> List | np.ndarray | torch.FloatTensor:
        assert self.page_geometry, "Cannot determine features without data of the entire page"
        PLACEHOLDER_VALUE = -100.0

        features = []

        width = self.get_width()
        height = self.get_height()

        #  1. Line width relative to it's neighbourhood average
        vertical_neighbourhood = self.get_vertical_neighbours(2)
        vertical_neighbourhood_avg_width = sum([l.get_width() for l in vertical_neighbourhood]) / len(vertical_neighbourhood)
        features.append(width / vertical_neighbourhood_avg_width)

        #  2. Line width relative to page average
        #  3. Line height relative to page average
        features.append(width / self.page_geometry.get_avg_line_width())
        features.append(height / self.page_geometry.get_avg_line_height())

        # 10. Line width relative to page width
        # 11. Line height relative to page height
        features.append(width / self.page_geometry.page_width)
        features.append(height / self.page_geometry.page_height)

        # 12. Center X coordinate relative to page width
        # 13. Center Y coordinate relative to page height
        features.append(self.center.x / self.page_geometry.page_width)
        features.append(self.center.y / self.page_geometry.page_height)

        # 22. Left X coordinate relative to page width
        # 23. Right X coordinate relative to page width
        # 24. Top Y coordinate relative to page height
        # 25. Bottom Y coordinate relative to page height
        features.append(self.bbox.xmin / self.page_geometry.page_width)
        features.append(self.bbox.xmax / self.page_geometry.page_width)
        features.append(self.bbox.ymin / self.page_geometry.page_height)
        features.append(self.bbox.ymax / self.page_geometry.page_height)

        # 21. Line width relative to line height
        features.append(width / height)

        # 18. Bounding box area relative to page area
        features.append(self.bbox_area / self.page_geometry.page_area)

        #  4. L2 distance from parent (center - center)
        #  6. L2 distance from parent (center - center) relative to my height
        # 14. Distance from parent in Y axis (center - center)
        # 16. Distance from parent in X axis (center - center)
        # 19. Bounding box area relative to parent bounding box area
        if self.parent:
            distance_to_parent = dist_l2(self.center, self.parent.center)
            features.append(distance_to_parent)
            features.append(distance_to_parent / height)
            features.append(abs(self.center.y - self.parent.center.y))
            features.append(abs(self.center.x - self.parent.center.x))
            features.append(self.bbox_area / self.parent.bbox_area)
        else:
            features.extend([PLACEHOLDER_VALUE] * 5)

        #  5. L2 distance from child (center - center)
        #  7. L2 distance from child (center - center) relative to my height
        # 15. Distance from child in Y axis (center - center)
        # 17. Distance from child in X axis (center - center)
        # 20. Bounding box area relative to child bounding box area
        if self.child:
            distance_to_child = dist_l2(self.center, self.child.center)
            features.append(distance_to_child)
            features.append(distance_to_child / height)
            features.append(abs(self.center.y - self.child.center.y))
            features.append(abs(self.center.x - self.child.center.x))
            features.append(self.bbox_area / self.child.bbox_area)
        else:
            features.extend([PLACEHOLDER_VALUE] * 5)

        #  8. Number of predecessors
        features.append(self.get_number_of_predecessors())

        #  9. Number of children
        features.append(self.get_number_of_successors())

        match return_type:
            case "pt":
                return torch.tensor(features, dtype=torch.float32)

            case "np":
                return np.array(features, dtype=np.float32)

            case _:
                return features