"""Conservative order and grouping proposals for a reviewed SVG scene."""


def suggest_overlap(first, second, first_area, second_area, pixels):
    overlap = {"layers": [first["id"], second["id"]], "intersection_pixels": pixels,
               "ordering": "unknown", "evidence": "mask intersection; not proof of occlusion"}
    if first["kind"] != "object" or second["kind"] != "object":
        return overlap
    smaller, larger = (first, second) if first_area < second_area else (second, first)
    small_area, big_area = sorted((first_area, second_area))
    if small_area <= big_area * .55 and pixels >= small_area * .72:
        overlap.update(ordering="suggested", behind=larger["id"], in_front=smaller["id"],
                       smaller_mask_fraction=round(pixels / small_area, 4),
                       evidence="smaller object mostly contained in larger mask; assumed in front")
    return overlap


def suggested_paint_order(layers, overlaps):
    baseline = sorted(layers, key=lambda layer: layer["kind"] != "object")
    behind = {layer["id"]: [] for layer in baseline}
    for overlap in overlaps:
        if overlap["ordering"] in {"suggested", "confirmed"}:
            behind[overlap["in_front"]].append(overlap["behind"])
    order = []
    placed, visiting = set(), set()

    def place(ident):
        if ident in placed:
            return
        if ident in visiting:
            raise ValueError("Layer ordering evidence contains a cycle")
        visiting.add(ident)
        for predecessor in behind[ident]:
            place(predecessor)
        visiting.remove(ident)
        placed.add(ident)
        order.append(ident)

    for layer in baseline:
        place(layer["id"])
    return order


def propose_groups(layers):
    guides = [item for item in layers if item.get("role") == "grid-line"]
    for guide in guides:
        guide["group"] = "grade"
    titles = [item for item in layers if item["kind"] == "text-block"]
    if len(titles) != 1:
        return
    title = titles[0]
    tx, ty, tw, th = title["bbox"]
    for item in layers:
        if item["kind"] != "text":
            continue
        x, y, _, height = item["bbox"]
        if tx+tw-8 <= x <= tx+tw+tw*.75 and y+height >= ty-12 and y <= ty+th+12 and height <= th*.34:
            item["group"] = "ficha-tecnica"
    for line in title["lines"]:
        x, y, width, height = line["bbox"]
        right, bottom = x+width, y+height
        for guide in guides:
            gx, gy, gw, gh = guide["bbox"]
            if guide.get("orientation") == "vertical" and abs(gx-right) <= 4 and gy < bottom and gy+gh > y:
                right = min(right, gx - 2)
            elif guide.get("orientation") == "horizontal" and abs(gy-bottom) <= 4 and gx < right and gx+gw > x:
                bottom = min(bottom, gy - 2)
        if right > x and bottom > y and (right-x != width or bottom-y != height):
            line["layout_bbox"] = [x, y, right-x, bottom-y]
