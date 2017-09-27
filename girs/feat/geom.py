from osgeo import ogr


# =============================================================================
# Point
# =============================================================================
def create_point(*args):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(*args)
    return point


def create_point25D(*args):
    point = ogr.Geometry(ogr.wkbPoint25D)
    point.AddPoint(*args)
    return point


def create_pointM(*args):
    point = ogr.Geometry(ogr.wkbPointM)
    point.AddPointM(*args)
    return point


def create_pointZM(*args):
    point = ogr.Geometry(ogr.wkbPointZM)
    point.AddPointZM(*args)
    return point


# =============================================================================
# MultiPoint
# =============================================================================
def create_multi_point(points):
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for point in points:
        p = ogr.Geometry(ogr.wkbPoint)
        p.AddPoint(*point)
        multipoint.AddGeometry(p)
    return multipoint


def create_multi_point25D(points):
    multipoint = ogr.Geometry(ogr.wkbMultiPoint25D)
    for point in points:
        p = ogr.Geometry(ogr.wkbPoint25D)
        p.AddPoint(*point)
        multipoint.AddGeometry(p)
    return multipoint


def create_multi_pointM(points):
    multipoint = ogr.Geometry(ogr.wkbMultiPointM)
    for point in points:
        p = ogr.Geometry(ogr.wkbPointM)
        p.AddPointM(*point)
        multipoint.AddGeometry(p)
    return multipoint


def create_multi_pointZM(points):
    multipoint = ogr.Geometry(ogr.wkbMultiPointZM)
    for point in points:
        p = ogr.Geometry(ogr.wkbPointZM)
        p.AddPointZM(*point)
        multipoint.AddGeometry(p)
    return multipoint


# =============================================================================
# LineString
# =============================================================================
def create_line_string(line_string):
    line = ogr.Geometry(ogr.wkbLineString)
    for p in line_string:
        line.AddPoint(*p)
    return line


def create_line_string25D(line_string):
    line = ogr.Geometry(ogr.wkbLineString25D)
    for p in line_string:
        line.AddPoint(*p)
    return line


def create_line_stringM(line_string):
    line = ogr.Geometry(ogr.wkbLineStringM)
    for p in line_string:
        line.AddPointM(*p)
    return line


def create_line_stringZM(line_string):
    line = ogr.Geometry(ogr.wkbLineStringZM)
    for p in line_string:
        line.AddPointZM(*p)
    return line


# =============================================================================
# MultiLineString
# =============================================================================
def create_multiline_string(line_strings):
    multiline = ogr.Geometry(ogr.wkbMultiLineString)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_string(line_string))
    return multiline


def create_multiline_string25D(line_strings):
    multiline = ogr.Geometry(ogr.wkbMultiLineString25D)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_string25D(line_string))
    return multiline


def create_multiline_stringM(line_strings):
    multiline = ogr.Geometry(ogr.wkbMultiLineStringM)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_stringM(line_string))
    return multiline


def create_multiline_stringZM(line_strings):
    multiline = ogr.Geometry(ogr.wkbMultiLineStringZM)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_stringZM(line_string))
    return multiline


# =============================================================================
# LineString
# =============================================================================
def create_linear_ring(points):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if points[0] != points[-1]:
        points.append(points[0])
    for p in points:
        ring.AddPoint(*p)
    return ring


def create_linear_ring2D(points):
    return create_linear_ring(points)


def create_linear_ringM(points):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if points[0] != points[-1]:
        points.append(points[0])
    for p in points:
        ring.AddPointM(*p)
    return ring


def create_linear_ringZM(points):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if points[0] != points[-1]:
        points.append(points[0])
    for p in points:
        ring.AddPointZM(*p)
    return ring


# =============================================================================
# Polygon
# =============================================================================
def create_polygon(rings):
    poly = ogr.Geometry(ogr.wkbPolygon)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ring(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ring(rings))
    return poly


def create_polygon25D(rings):
    poly = ogr.Geometry(ogr.wkbPolygon25D)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ring2D(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ring2D(rings))
    return poly


def create_polygonM(rings):
    poly = ogr.Geometry(ogr.wkbPolygonM)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ringM(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ringM(rings))
    return poly


def create_polygonZM(rings):
    poly = ogr.Geometry(ogr.wkbPolygonZM)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ringZM(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ringZM(rings))
    return poly


# =============================================================================
# Polygon
# =============================================================================
def create_multipolygon(polys):
    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)
    try:
        for rings in polys:
            multipoly.AddGeometry(create_polygon(rings))
    except (TypeError, AttributeError):
        multipoly.AddGeometry(create_polygon(polys))
    return multipoly


def create_multipolygon25D(polys):
    multipoly = ogr.Geometry(ogr.wkbMultiPolygon25D)
    try:
        for rings in polys:
            multipoly.AddGeometry(create_polygon25D(rings))
    except (TypeError, AttributeError):
        multipoly.AddGeometry(create_polygon25D(polys))
    return multipoly


def create_multipolygonM(polys):
    multipoly = ogr.Geometry(ogr.wkbMultiPolygonM)
    try:
        for rings in polys:
            multipoly.AddGeometry(create_polygonM(rings))
    except (TypeError, AttributeError):
        multipoly.AddGeometry(create_polygonM(polys))
    return multipoly


def create_multipolygonZM(polys):
    multipoly = ogr.Geometry(ogr.wkbMultiPolygonZM)
    try:
        for rings in polys:
            multipoly.AddGeometry(create_polygonZM(rings))
    except (TypeError, AttributeError):
        multipoly.AddGeometry(create_polygonZM(polys))
    return multipoly


if __name__ == '__main__':
    p_list0 = [(0.00, 0.00), (1.00, 0.00), (1.00, 1.00), (0.00, 1.00)]
    p_list1 = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    p_list2 = [(p[0]+1, p[1]) for p in p_list0]
    p_list3 = [(p[0]+1, p[1]) for p in p_list1]
    p_list0_25D = [(0.00, 0.00, 0.50), (1.00, 0.00, 0.50), (1.00, 1.00, 0.50), (0.00, 1.00, 0.50)]
    p_list1_25D = [(0.25, 0.25, 0.75), (0.75, 0.25, 0.75), (0.75, 0.75, 0.75), (0.25, 0.75, 0.75)]
    p_list2_25D = [(p[0]+1, p[1], p[2]) for p in p_list0_25D]
    p_list3_25D = [(p[0]+1, p[1], p[2]) for p in p_list1_25D]
    p_list0_ZM = [(0.00, 0.00, 0.50, 0.25), (1.00, 0.00, 0.50, 0.25), (1.00, 1.00, 0.50, 0.25), (0.00, 1.00, 0.50, 0.25)]
    p_list1_ZM = [(0.25, 0.25, 0.75, 0.25), (0.75, 0.25, 0.75, 0.25), (0.75, 0.75, 0.75, 0.25), (0.25, 0.75, 0.75, 0.25)]
    p_list2_ZM = [(p[0]+1, p[1], p[2], p[3]) for p in p_list0_ZM]
    p_list3_ZM = [(p[0]+1, p[1], p[2], p[3]) for p in p_list1_ZM]

    print create_multipolygon(p_list0).ExportToWkt()
    print create_multipolygon([p_list0, p_list2]).ExportToWkt()
    print create_multipolygon([[p_list0, p_list1], p_list2]).ExportToWkt()
    print create_multipolygon([[p_list0, p_list1], [p_list2, p_list3]]).ExportToWkt()

    print create_multipolygon25D(p_list0_25D).ExportToWkt()
    print create_multipolygon25D([p_list0_25D, p_list2_25D]).ExportToWkt()
    print create_multipolygon25D([[p_list0_25D, p_list1_25D], p_list2_25D]).ExportToWkt()
    print create_multipolygon25D([[p_list0_25D, p_list1_25D], [p_list2_25D, p_list3_25D]]).ExportToWkt()

    print create_multipolygonM(p_list0_25D).ExportToWkt()
    print create_multipolygonM([p_list0_25D, p_list2_25D]).ExportToWkt()
    print create_multipolygonM([[p_list0_25D, p_list1_25D], p_list2_25D]).ExportToWkt()
    print create_multipolygonM([[p_list0_25D, p_list1_25D], [p_list2_25D, p_list3_25D]]).ExportToWkt()

    print create_multipolygonZM(p_list0_ZM).ExportToWkt()
    print create_multipolygonZM([p_list0_ZM, p_list2_ZM]).ExportToWkt()
    print create_multipolygonZM([[p_list0_ZM, p_list1_ZM], p_list2_ZM]).ExportToWkt()
    print create_multipolygonZM([[p_list0_ZM, p_list1_ZM], [p_list2_ZM, p_list3_ZM]]).ExportToWkt()
