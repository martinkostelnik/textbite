import numpy as np
import lxml.etree as ET


def get_covering_bites(lines_to_cover, bites):
    lines_to_cover = set(lines_to_cover)
    regions = []

    while lines_to_cover:
        l_to_cover = next(iter(lines_to_cover))
        r_id, r = [(i, b) for i, b in enumerate(bites) if l_to_cover in b][0]
        r = set(r)
        clean = r.issubset(lines_to_cover)
        lines_to_cover -= r
        regions.append((r_id, clean))

    return regions


# https://stackoverflow.com/a/66801704/9703830
def polygon_area(xs, ys):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    # https://stackoverflow.com/a/30408825/7128154
    return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def polygon_centroid(xs, ys):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    xy = np.array([xs, ys])
    c = np.dot(xy + np.roll(xy, 1, axis=1),
               xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
               ) / (6 * polygon_area(xs, ys))
    return c


def get_bite_reading_order(region_centers, bite_regions):
    y_sorted_regions = sorted(bite_regions, key=lambda r: region_centers[r][1])

    # get significantly y-overlapped regions
    # and sort them by x-axis (often a split header line or similar)
    yx_sorted_regions = []

    running_y_overlapped = None
    Y_MARGIN = 5  # if two regions are not at least this many y-pixels aside, consider them overlapping
    for r in y_sorted_regions:
        if running_y_overlapped is None:
            running_y_overlapped = [r]
            continue

        if region_centers[r][1] < region_centers[running_y_overlapped[-1]][1] + Y_MARGIN:
            running_y_overlapped.append(r)
        else:
            yx_sorted_regions.extend(sorted(running_y_overlapped, key=lambda r: region_centers[r][0]))
            running_y_overlapped = [r]

    yx_sorted_regions.extend(sorted(running_y_overlapped, key=lambda r: region_centers[r]))

    bite_reading_order = [r for r in yx_sorted_regions]

    print('Bite!')
    for r in bite_reading_order:
        print(region_centers[r])

    assert len(bite_regions) == len(bite_reading_order)
    return bite_reading_order


def process(layout, bites):
    layout_bites = [[line.id for line in r.lines] for r in layout.regions]
    all_lines_ids = list(line.id for line in layout.lines_iterator())
    out_of_bite_lines = set(all_lines_ids) - set(sum((b.lines for b in bites), []))
    region_centers = [polygon_centroid(*zip(*r.polygon)) for r in layout.regions]

    assert len(set(sum((b.lines for b in bites), []))) == sum(len(b.lines) for b in bites), 'Bites have to be disjunct'
    assert set(sum((b.lines for b in bites), [])).issubset(all_lines_ids), 'Bites must only have lines from the layout'

    coverage = [get_covering_bites(bite.lines, layout_bites) for bite in bites]

    # if there are un-pure regions, split them
    #   to make them pure
    #   create new polygons by surrounding the textlines (like done in producing annotation for YOLO)
    for bite in coverage:
        for r in bite:
            assert r[1] is True, 'Unpure regions unsupported yet!'

    reading_order = []
    for bite in coverage:
        reading_order.append(get_bite_reading_order(region_centers, [r[0] for r in bite]))

    out_of_bite_regions = get_covering_bites(out_of_bite_lines, layout_bites)

    # create XML reading order
    reading_order_root = ET.Element("ReadingOrder")
    unordered_root = ET.SubElement(reading_order_root, "UnorderedGroup", attrib={'id': 'root'})
    for bite_id, bite in enumerate(reading_order):
        xml_bite = ET.SubElement(unordered_root, "OrderedGroup", attrib={'id': f'bite_{bite_id+1}'})
        for i, region_id in enumerate(bite):
            ET.SubElement(xml_bite, 'RegionRefIndexed', attrib={'regionRef': layout.regions[region_id].id, 'index': str(i)})

    # get xml (layout -> string -> XML)
    out_xml = ET.fromstring(layout.to_pagexml_string())

    # inject the reading order
    ns = out_xml.nsmap[None]
    page_qname = ET.QName(ns, 'Page')
    out_xml.find(page_qname).insert(0, reading_order_root)
    ET.indent(out_xml)

    return ET.tostring(out_xml, pretty_print=True, encoding=str)
