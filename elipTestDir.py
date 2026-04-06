import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIG
# =========================
IMG_NAME = "1.png"
ROOT_PATH = "img"
OUTPUT_DIR = "output"

FLOWER_CENTER_MODE = "ellipse"
PISTIL_PERCENTILE = 80
CENTER_RADIUS_RATIO = 0.12

FLOWER_MIN_COMPONENT_AREA = 800
MINOR_AXIS_DIST_DELTA = 3.0

LAB_PATCH_RADIUS = 80
VECTOR_SCALE = 2.0


# =========================
# BASIC HELPERS
# =========================
def apply_mask_to_bgr(img_bgr, mask):
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def normalize_to_uint8(x):
    x = x.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)

    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)

    y = (x - x_min) / (x_max - x_min)
    return np.clip(255.0 * y, 0, 255).astype(np.uint8)


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_id, 255, 0).astype(np.uint8)


def remove_small_components(mask, min_area=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def largest_contour_from_mask(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def contour_centroid_from_contour(cnt):
    if cnt is None:
        return None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-9:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def project_point_to_line(pt, line_point, line_dir):
    if pt is None or line_point is None or line_dir is None:
        return None

    p = np.asarray(pt, dtype=np.float32)
    p0 = np.asarray(line_point, dtype=np.float32)
    d = np.asarray(line_dir, dtype=np.float32)

    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None

    d = d / n
    t = float(np.dot(p - p0, d))
    proj = p0 + t * d
    return proj


def point_to_line_distance(pt, line_point, line_dir):
    proj = project_point_to_line(pt, line_point, line_dir)
    if proj is None:
        return None
    return float(np.linalg.norm(np.asarray(pt, dtype=np.float32) - proj))


def contour_to_points_list(cnt):
    if cnt is None:
        return []
    pts = np.asarray(cnt).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in pts]


def mask_to_points_list(mask):
    ys, xs = np.where(mask > 0)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def write_points_block(f, name, points, points_per_line=12):
    f.write(f"{name}_count = {len(points)}\n")
    f.write(f"{name} = [\n")
    for i in range(0, len(points), points_per_line):
        chunk = points[i:i + points_per_line]
        line = ", ".join([f"({x}, {y})" for x, y in chunk])
        f.write(f"  {line},\n")
    f.write("]\n\n")


def contour_filled_mask(cnt, shape_hw):
    h, w = shape_hw[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if cnt is not None:
        cv2.drawContours(out, [cnt], -1, 255, thickness=-1)
    return out


def iou_binary_masks(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def save_geometry_to_txt(
    txt_path,
    img_name,
    flower_mask,
    flower_cnt,
    flower_geom_info,
    flower_minor_info,
    pistil_mask,
    pistil_cnt,
    pistil_axis_info,
):
    flower_contour_points = contour_to_points_list(flower_cnt)
    pistil_contour_points = contour_to_points_list(pistil_cnt)
    pistil_mask_points = mask_to_points_list(pistil_mask)

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"image_name = {img_name}\n\n")

        f.write("========== FLOWER ==========\n")
        f.write(f"flower_mask_nonzero_count = {int(np.count_nonzero(flower_mask))}\n\n")
        write_points_block(f, "flower_mask_contour_points", flower_contour_points)

        flower_mode = flower_geom_info.get("mode", None)
        f.write(f"flower_geometry_mode = {flower_mode}\n")
        if flower_mode == "ellipse":
            (cx, cy), (MA, ma), angle = flower_geom_info["ellipse"]
            f.write(f"flower_ellipse_center = ({cx:.6f}, {cy:.6f})\n")
            f.write(f"flower_ellipse_axes = (major_like={MA:.6f}, minor_like={ma:.6f})\n")
            f.write(f"flower_ellipse_angle_deg = {angle:.6f}\n")
        elif flower_mode == "circle":
            (cx, cy), radius = flower_geom_info["circle"]
            f.write(f"flower_circle_center = ({cx:.6f}, {cy:.6f})\n")
            f.write(f"flower_circle_radius = {radius:.6f}\n")
        f.write("\n")

        if flower_minor_info:
            f.write("flower_minor_axis_available = True\n")
            f.write(f"flower_minor_axis_center = {flower_minor_info.get('center_int', None)}\n")
            f.write(f"flower_minor_axis_length = {flower_minor_info.get('minor_len', None)}\n")
            f.write(f"flower_minor_axis_dir = {flower_minor_info.get('minor_dir', None)}\n")
            f.write(f"flower_minor_axis_point_1 = {flower_minor_info.get('minor_p1', None)}\n")
            f.write(f"flower_minor_axis_point_2 = {flower_minor_info.get('minor_p2', None)}\n")
            f.write(f"flower_major_axis_point_1 = {flower_minor_info.get('major_p1', None)}\n")
            f.write(f"flower_major_axis_point_2 = {flower_minor_info.get('major_p2', None)}\n")
        else:
            f.write("flower_minor_axis_available = False\n")
        f.write("\n")

        f.write("========== PISTIL ==========\n")
        f.write(f"pistil_mask_nonzero_count = {int(np.count_nonzero(pistil_mask))}\n\n")
        write_points_block(f, "pistil_mask_nonzero_points", pistil_mask_points)
        write_points_block(f, "pistil_mask_contour_points", pistil_contour_points)

        f.write(f"pistil_mode = {pistil_axis_info.get('mode', None)}\n")
        f.write(f"pistil_shape_selected = {pistil_axis_info.get('shape_selected', None)}\n")
        f.write(f"pistil_circle_iou = {pistil_axis_info.get('circle_iou', None)}\n")
        f.write(f"pistil_ellipse_iou = {pistil_axis_info.get('ellipse_iou', None)}\n")
        f.write(f"pistil_selected_by = {pistil_axis_info.get('selected_by', None)}\n")

        circle = pistil_axis_info.get("circle", None)
        if circle is not None:
            (ccx, ccy), cr = circle
            f.write(f"pistil_circle_center = ({ccx:.6f}, {ccy:.6f})\n")
            f.write(f"pistil_circle_radius = {cr:.6f}\n")

        ellipse = pistil_axis_info.get("ellipse", None)
        if ellipse is not None:
            (ecx, ecy), (ea, eb), eangle = ellipse
            f.write(f"pistil_ellipse_center = ({ecx:.6f}, {ecy:.6f})\n")
            f.write(f"pistil_ellipse_axes = (axis_a={ea:.6f}, axis_b={eb:.6f})\n")
            f.write(f"pistil_ellipse_angle_deg = {eangle:.6f}\n")
            f.write(f"pistil_ellipse_major_len = {pistil_axis_info.get('ellipse_major_len', None)}\n")
            f.write(f"pistil_ellipse_minor_len = {pistil_axis_info.get('ellipse_minor_len', None)}\n")

        f.write(f"pistil_line_candidate_a = {pistil_axis_info.get('candidate_a', None)}\n")
        f.write(f"pistil_line_candidate_b = {pistil_axis_info.get('candidate_b', None)}\n")
        f.write(f"pistil_selected_base_point = {pistil_axis_info.get('base_pt', None)}\n")
        f.write(f"pistil_selected_tip_point = {pistil_axis_info.get('tip_pt', None)}\n")
        f.write(f"L_candidate_a = {pistil_axis_info.get('L_candidate_a', None)}\n")
        f.write(f"L_candidate_b = {pistil_axis_info.get('L_candidate_b', None)}\n")
        f.write(f"dist_a_to_minor_axis = {pistil_axis_info.get('dist_a_to_minor_axis', None)}\n")
        f.write(f"dist_b_to_minor_axis = {pistil_axis_info.get('dist_b_to_minor_axis', None)}\n")
        f.write(f"dist_gap = {pistil_axis_info.get('dist_gap', None)}\n")
        f.write(f"axis_dist_delta = {pistil_axis_info.get('axis_dist_delta', None)}\n")
        f.write(f"minor_axis_available_for_selection = {pistil_axis_info.get('minor_axis_available', None)}\n")


# =========================
# FLOWER MASK
# =========================
def flower_mask_hsv(img_bgr, min_component_area=800):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([18, 80, 30], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask = remove_small_components(mask, min_area=min_component_area)

    return mask


# =========================
# PISTIL MASK
# =========================
def keep_component_nearest_center(mask, cx, cy, min_area=3):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    best_id = -1
    best_dist = 1e18

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x_i, y_i = centroids[i]
        d2 = (x_i - cx) ** 2 + (y_i - cy) ** 2
        if d2 < best_dist:
            best_dist = d2
            best_id = i

    if best_id == -1:
        return np.zeros_like(mask)

    return np.where(labels == best_id, 255, 0).astype(np.uint8)

def pistil_mask_from_flower_only_rgb(
    flower_only_bgr,
    flower_mask,
    center_radius_ratio=0.22,
    r_min=155,
    r_max=180,
    g_min=155,
    g_max=180,
):
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner_flower = cv2.erode(flower_mask, kernel_inner, iterations=1)

    ys, xs = np.where(inner_flower > 0)
    if xs.size == 0:
        return np.zeros_like(flower_mask), np.zeros_like(flower_mask)

    M = cv2.moments(inner_flower)
    if abs(M["m00"]) > 1e-9:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

    x0, x1 = int(np.min(xs)), int(np.max(xs))
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1

    h, w = flower_mask.shape
    Y, X = np.ogrid[:h, :w]
    r_limit = max(12, int(center_radius_ratio * max(box_w, box_h)))
    center_gate = ((X - cx) ** 2 + (Y - cy) ** 2) <= r_limit ** 2

    valid = (inner_flower > 0) & center_gate

    rgb = cv2.cvtColor(flower_only_bgr, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(rgb)

    candidate = (
        (R >= r_min) & (R <= r_max) &
        (G >= g_min) & (G <= g_max)
    )

    pistil = np.zeros_like(flower_mask)
    pistil[candidate & valid] = 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    pistil = cv2.morphologyEx(pistil, cv2.MORPH_OPEN, kernel_small, iterations=1)
    pistil = cv2.morphologyEx(pistil, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    pistil = keep_largest_component(pistil)

    debug_vis = np.zeros_like(flower_mask)
    debug_vis[valid] = 90
    debug_vis[candidate & valid] = 255

    return pistil, debug_vis

def pistil_mask_from_flower_only(
    flower_only_bgr,
    flower_mask,
    pistil_percentile=80,
    center_radius_ratio=0.22,
):
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    inner_flower = cv2.erode(flower_mask, kernel_inner, iterations=1)

    ys, xs = np.where(inner_flower > 0)
    if xs.size == 0:
        return np.zeros_like(flower_mask), np.zeros_like(flower_mask)

    M = cv2.moments(inner_flower)
    if abs(M["m00"]) > 1e-9:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

    x0, x1 = int(np.min(xs)), int(np.max(xs))
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1

    h, w = flower_mask.shape
    Y, X = np.ogrid[:h, :w]
    r_limit = max(12, int(center_radius_ratio * max(box_w, box_h)))
    center_gate = ((X - cx) ** 2 + (Y - cy) ** 2) <= r_limit ** 2

    b, g, r = cv2.split(flower_only_bgr.astype(np.float32))
    score = (g - r) + 0.01 * (g - b)
    # score = 2 * g - r - b

    valid = (inner_flower > 0) & center_gate
    vals = score[valid]
    if vals.size == 0:
        return np.zeros_like(flower_mask), np.zeros_like(flower_mask)

    thr = np.percentile(vals, pistil_percentile)

    pistil = np.zeros_like(flower_mask)

    pistil[(score >= thr) & valid] = 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    pistil = cv2.morphologyEx(pistil, cv2.MORPH_OPEN, kernel_small, iterations=1)
    pistil = cv2.morphologyEx(pistil, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    pistil = keep_largest_component(pistil)
    score_vis = normalize_to_uint8(score)
    return pistil, score_vis


# =========================
# FLOWER GEOMETRY
# =========================
def flower_reference_center_from_outer_contour(cnt, mode="ellipse"):
    if cnt is None:
        return None, {}

    info = {}

    if mode == "ellipse" and len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        center = (int(round(cx)), int(round(cy)))
        info["mode"] = "ellipse"
        info["ellipse"] = ((cx, cy), (MA, ma), angle)
        return center, info

    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    center = (int(round(cx)), int(round(cy)))
    info["mode"] = "circle"
    info["circle"] = ((cx, cy), radius)
    return center, info


def flower_minor_axis_from_contour(cnt):
    if cnt is None or len(cnt) < 5:
        return {}

    (cx, cy), (axis_a, axis_b), angle_deg = cv2.fitEllipse(cnt)
    center = np.array([cx, cy], dtype=np.float32)
    theta = np.deg2rad(angle_deg)

    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

    if axis_a >= axis_b:
        major_len = float(axis_a)
        minor_len = float(axis_b)
        major_dir = u / max(np.linalg.norm(u), 1e-12)
        minor_dir = v / max(np.linalg.norm(v), 1e-12)
    else:
        major_len = float(axis_b)
        minor_len = float(axis_a)
        major_dir = v / max(np.linalg.norm(v), 1e-12)
        minor_dir = u / max(np.linalg.norm(u), 1e-12)

    major_p1 = center - 0.5 * major_len * major_dir
    major_p2 = center + 0.5 * major_len * major_dir
    minor_p1 = center - 0.5 * minor_len * minor_dir
    minor_p2 = center + 0.5 * minor_len * minor_dir

    return {
        "center": (float(cx), float(cy)),
        "center_int": (int(round(cx)), int(round(cy))),
        "ellipse": ((cx, cy), (axis_a, axis_b), angle_deg),
        "major_len": major_len,
        "minor_len": minor_len,
        "major_dir": major_dir,
        "minor_dir": minor_dir,
        "major_p1": tuple(np.round(major_p1).astype(int)),
        "major_p2": tuple(np.round(major_p2).astype(int)),
        "minor_p1": tuple(np.round(minor_p1).astype(int)),
        "minor_p2": tuple(np.round(minor_p2).astype(int)),
    }


# =========================
# PISTIL BASE/TIP
# =========================
def pistil_rotated_rect_from_contour(cnt):
    if cnt is None or len(cnt) < 3:
        return None, None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(int)
    return rect, box


def pistil_ellipse_from_contour(cnt):
    if cnt is None or len(cnt) < 5:
        return None
    return cv2.fitEllipse(cnt)


def draw_circle_mask(circle, shape_hw):
    h, w = shape_hw[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if circle is None:
        return out
    (cx, cy), r = circle
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), 255, thickness=-1)
    return out


def draw_ellipse_mask(ellipse, shape_hw):
    h, w = shape_hw[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if ellipse is None:
        return out
    cv2.ellipse(out, ellipse, 255, thickness=-1)
    return out


def mean_L_around_point(L_channel, pt, radius=2):
    if L_channel is None or pt is None:
        return None

    x, y = pt
    h, w = L_channel.shape[:2]

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return None

    patch = L_channel[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    return float(np.mean(patch))


def endpoints_from_rotated_rect(rect, box):
    if rect is None or box is None:
        return None, None, {}

    (cx, cy), (w, h), angle = rect
    center = np.array([cx, cy], dtype=np.float32)
    center_int = (int(round(cx)), int(round(cy)))

    box_f = box.astype(np.float32)
    edges = []
    for i in range(4):
        p1 = box_f[i]
        p2 = box_f[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = float(np.linalg.norm(edge_vec))
        edges.append({"vec": edge_vec, "len": edge_len})

    long_edge = max(edges, key=lambda e: e["len"])
    d = long_edge["vec"]
    d_norm = float(np.linalg.norm(d))

    if d_norm < 1e-12:
        c = (int(round(cx)), int(round(cy)))
        return c, c, {
            "center": (float(cx), float(cy)),
            "center_int": c,
            "size": (float(w), float(h)),
            "angle": float(angle),
        }

    d = d / d_norm
    half_long = max(w, h) / 2.0

    p_a = center - half_long * d
    p_b = center + half_long * d

    p_a_int = tuple(np.round(p_a).astype(int))
    p_b_int = tuple(np.round(p_b).astype(int))

    info = {
        "center": (float(cx), float(cy)),
        "center_int": center_int,
        "size": (float(w), float(h)),
        "angle": float(angle),
    }
    return p_a_int, p_b_int, info


def endpoints_from_ellipse(ellipse):
    if ellipse is None:
        return None, None, {}

    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    center = np.array([cx, cy], dtype=np.float32)
    theta = np.deg2rad(angle_deg)

    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

    if axis_a >= axis_b:
        major_len = float(axis_a)
        minor_len = float(axis_b)
        major_dir = u / max(np.linalg.norm(u), 1e-12)
    else:
        major_len = float(axis_b)
        minor_len = float(axis_a)
        major_dir = v / max(np.linalg.norm(v), 1e-12)

    p_a = center - 0.5 * major_len * major_dir
    p_b = center + 0.5 * major_len * major_dir

    p_a_int = tuple(np.round(p_a).astype(int))
    p_b_int = tuple(np.round(p_b).astype(int))

    info = {
        "center": (float(cx), float(cy)),
        "center_int": (int(round(cx)), int(round(cy))),
        "ellipse_major_len": major_len,
        "ellipse_minor_len": minor_len,
        "ellipse_major_dir": major_dir,
    }
    return p_a_int, p_b_int, info


def select_base_tip_from_candidates(
    p_a_int,
    p_b_int,
    flower_center,
    flower_minor_info=None,
    L_channel=None,
    axis_dist_delta=3.0,
    lab_patch_radius=30,
):
    if p_a_int is None or p_b_int is None:
        return None, None, {}

    La = mean_L_around_point(L_channel, p_a_int, radius=lab_patch_radius)
    Lb = mean_L_around_point(L_channel, p_b_int, radius=lab_patch_radius)

    axis_point = None
    axis_dir = None
    dist_a = None
    dist_b = None
    dist_gap = None
    minor_axis_available = False

    if flower_minor_info:
        axis_point = flower_minor_info.get("center", None)
        axis_dir = flower_minor_info.get("minor_dir", None)
        dist_a = point_to_line_distance(p_a_int, axis_point, axis_dir)
        dist_b = point_to_line_distance(p_b_int, axis_point, axis_dir)

        if dist_a is not None and dist_b is not None:
            minor_axis_available = True
            dist_gap = abs(dist_a - dist_b)

    if minor_axis_available and dist_gap is not None and dist_gap > axis_dist_delta:
        if dist_a <= dist_b:
            base_pt = p_a_int
            tip_pt = p_b_int
        else:
            base_pt = p_b_int
            tip_pt = p_a_int
        selected_by = "minor_axis_distance"

    elif La is not None and Lb is not None:
        if La <= Lb:
            base_pt = p_a_int
            tip_pt = p_b_int
        else:
            base_pt = p_b_int
            tip_pt = p_a_int
        selected_by = "LAB_brightness"

    else:
        fc = np.array(flower_center, dtype=np.float32)
        pa = np.array(p_a_int, dtype=np.float32)
        pb = np.array(p_b_int, dtype=np.float32)
        da = float(np.linalg.norm(pa - fc))
        db = float(np.linalg.norm(pb - fc))

        if da <= db:
            base_pt = p_a_int
            tip_pt = p_b_int
        else:
            base_pt = p_b_int
            tip_pt = p_a_int
        selected_by = "distance_to_flower_center_fallback"

    return base_pt, tip_pt, {
        "candidate_a": p_a_int,
        "candidate_b": p_b_int,
        "L_candidate_a": La,
        "L_candidate_b": Lb,
        "minor_axis_point": axis_point,
        "minor_axis_dir": axis_dir,
        "dist_a_to_minor_axis": dist_a,
        "dist_b_to_minor_axis": dist_b,
        "dist_gap": dist_gap,
        "axis_dist_delta": float(axis_dist_delta),
        "minor_axis_available": bool(minor_axis_available),
        "selected_by": selected_by,
        "base_pt": base_pt,
        "tip_pt": tip_pt,
    }


def pistil_base_tip_by_iou(
    cnt,
    flower_center,
    flower_minor_info=None,
    L_channel=None,
    axis_dist_delta=3.0,
    lab_patch_radius=30,
    image_shape=None,
):
    if cnt is None or flower_center is None or image_shape is None:
        return None, None, {}

    object_mask = contour_filled_mask(cnt, image_shape)

    ellipse = pistil_ellipse_from_contour(cnt)
    ellipse_mask = draw_ellipse_mask(ellipse, image_shape) if ellipse is not None else np.zeros_like(object_mask)
    ellipse_iou = iou_binary_masks(object_mask, ellipse_mask) if ellipse is not None else -1.0

    circle = None
    circle_iou = -1.0
    circle_data = pistil_circle_from_contour(cnt)
    if circle_data is not None:
        (ccx, ccy), cr = circle_data
        circle = ((float(ccx), float(ccy)), float(cr))
        circle_mask = draw_circle_mask(circle, image_shape)
        circle_iou = iou_binary_masks(object_mask, circle_mask)

    if circle_iou >= ellipse_iou and circle is not None:
        shape_selected = "circle"
        center_pt = (int(round(circle[0][0])), int(round(circle[0][1])))
        info = {
            "mode": "circle_like",
            "shape_selected": shape_selected,
            "circle": circle,
            "ellipse": ellipse,
            "circle_iou": circle_iou,
            "ellipse_iou": ellipse_iou,
            "shape_selected_by": "IoU",
            "center": center_pt,
            "center_int": center_pt,
            "selected_by": "circle_center",
            "base_pt": center_pt,
            "tip_pt": center_pt,
            "candidate_a": None,
            "candidate_b": None,
            "L_candidate_a": None,
            "L_candidate_b": None,
            "minor_axis_point": None,
            "minor_axis_dir": None,
            "dist_a_to_minor_axis": None,
            "dist_b_to_minor_axis": None,
            "dist_gap": None,
            "axis_dist_delta": None,
            "minor_axis_available": False,
        }
        return center_pt, center_pt, info

    if ellipse is not None:
        shape_selected = "ellipse"
        p_a, p_b, geom_info = endpoints_from_ellipse(ellipse)
        base_pt, tip_pt, select_info = select_base_tip_from_candidates(
            p_a,
            p_b,
            flower_center,
            flower_minor_info=flower_minor_info,
            L_channel=L_channel,
            axis_dist_delta=axis_dist_delta,
            lab_patch_radius=lab_patch_radius,
        )
        info = {
            "mode": shape_selected,
            "shape_selected": shape_selected,
            "circle": circle,
            "ellipse": ellipse,
            "circle_iou": circle_iou,
            "ellipse_iou": ellipse_iou,
            "shape_selected_by": "IoU",
        }
        info.update(geom_info)
        info.update(select_info)
        return base_pt, tip_pt, info

    c = contour_centroid_from_contour(cnt)
    return c, c, {"mode": "centroid_fallback", "shape_selected": "centroid", "circle_iou": circle_iou, "ellipse_iou": ellipse_iou}


def pistil_circle_from_contour(cnt):
    if cnt is None:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return (int(round(cx)), int(round(cy))), int(round(r))


# =========================
# DIRECTION
# =========================
def direction_from_points(p_from, p_to):
    if p_from is None or p_to is None:
        return None

    x1, y1 = p_from
    x2, y2 = p_to

    dx = float(x2 - x1)
    dy = float(y2 - y1)
    norm = float(np.hypot(dx, dy))
    if norm < 1e-9:
        return None

    ux = dx / norm
    uy = dy / norm
    theta_math_deg = float(np.degrees(np.arctan2(-dy, dx)))
    theta_axis_deg = theta_math_deg % 180.0

    return {
        "dx": dx,
        "dy": dy,
        "norm": norm,
        "ux": ux,
        "uy": uy,
        "theta_math_deg": theta_math_deg,
        "theta_axis_deg": float(theta_axis_deg),
    }


# =========================
# DRAWING
# =========================
def draw_detection_image(
    img_bgr,
    flower_cnt,
    pistil_cnt,
    flower_center,
    pistil_base,
    pistil_tip,
    flower_geom_info,
    flower_minor_info,
    pistil_axis_info,
):
    out = img_bgr.copy()
    pistil_mode = pistil_axis_info.get("mode", "")

    if flower_cnt is not None:
        cv2.drawContours(out, [flower_cnt], -1, (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(flower_cnt)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if flower_geom_info.get("mode") == "circle":
        (cx, cy), r = flower_geom_info["circle"]
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 1)
    elif flower_geom_info.get("mode") == "ellipse":
        cv2.ellipse(out, flower_geom_info["ellipse"], (0, 255, 0), 1)

    if flower_center is not None:
        cv2.circle(out, flower_center, 6, (0, 255, 0), -1)

    if flower_minor_info:
        minor_p1 = flower_minor_info.get("minor_p1", None)
        minor_p2 = flower_minor_info.get("minor_p2", None)
        if minor_p1 is not None and minor_p2 is not None:
            cv2.line(out, minor_p1, minor_p2, (255, 0, 255), 2)

    if pistil_cnt is not None:
        cv2.drawContours(out, [pistil_cnt], -1, (0, 0, 255), 2)

        ellipse = pistil_axis_info.get("ellipse", None)
        if ellipse is not None:
            cv2.ellipse(out, ellipse, (255, 255, 0), 2)

        circle = pistil_axis_info.get("circle", None)
        if circle is not None:
            (ccx, ccy), cr = circle
            cv2.circle(out, (int(round(ccx)), int(round(ccy))), int(round(cr)), (0, 165, 255), 2)

        shape_selected = pistil_axis_info.get("shape_selected", None)
        if shape_selected == "ellipse" and ellipse is not None:
            cv2.putText(out, "IoU -> ellipse", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
        elif shape_selected == "circle" and circle is not None:
            cv2.putText(out, "IoU -> circle", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)

    if pistil_mode == "circle_like":
        center_pt = pistil_axis_info.get("center", pistil_base)
        if center_pt is not None:
            cv2.circle(out, center_pt, 6, (255, 255, 0), -1)
            if flower_center is not None:
                cv2.line(out, center_pt, flower_center, (0, 255, 0), 1)
        return out

    if pistil_base is not None:
        cv2.circle(out, pistil_base, 6, (255, 255, 0), -1)

    if pistil_tip is not None:
        cv2.circle(out, pistil_tip, 6, (0, 255, 255), -1)

    if pistil_base is not None and pistil_tip is not None:
        cv2.line(out, pistil_base, pistil_tip, (255, 255, 255), 2)

    if flower_center is not None and pistil_base is not None:
        cv2.line(out, pistil_base, flower_center, (0, 255, 0), 1)

    return out


def draw_direction_vector_image(
    img_bgr,
    flower_cnt,
    pistil_cnt,
    flower_center,
    pistil_base,
    pistil_tip,
    pistil_axis_info,
    vector_scale=12.0,
):
    out = img_bgr.copy()
    pistil_mode = pistil_axis_info.get("mode", "")

    if flower_cnt is not None:
        cv2.drawContours(out, [flower_cnt], -1, (255, 0, 0), 2)

    if pistil_cnt is not None:
        cv2.drawContours(out, [pistil_cnt], -1, (0, 0, 255), 2)

    if flower_center is not None:
        cv2.circle(out, flower_center, 6, (0, 255, 0), -1)
        cv2.putText(out, f"Flower C = {flower_center}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

    shape_selected = pistil_axis_info.get("shape_selected", "unknown")
    circle_iou = pistil_axis_info.get("circle_iou", None)
    ellipse_iou = pistil_axis_info.get("ellipse_iou", None)
    cv2.putText(out, f"shape = {shape_selected}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    if circle_iou is not None and ellipse_iou is not None:
        cv2.putText(out, f"IoU circle = {circle_iou:.3f} | IoU ellipse = {ellipse_iou:.3f}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    if pistil_mode == "circle_like":
        center_pt = pistil_axis_info.get("center", pistil_base)
        if center_pt is not None:
            cv2.circle(out, center_pt, 6, (255, 255, 0), -1)
            cv2.putText(out, f"Pistil center = {center_pt}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2, cv2.LINE_AA)
            if flower_center is not None:
                cv2.line(out, center_pt, flower_center, (0, 255, 0), 2)
        cv2.putText(out, "Circle-like pistil -> chi xet tam, khong ve vector huong", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        return out, None

    if pistil_base is not None:
        cv2.circle(out, pistil_base, 6, (255, 255, 0), -1)
    if pistil_tip is not None:
        cv2.circle(out, pistil_tip, 6, (0, 255, 255), -1)

    if pistil_base is not None and flower_center is not None:
        cv2.line(out, pistil_base, flower_center, (0, 255, 0), 2)

    if pistil_base is not None and pistil_tip is not None:
        cv2.line(out, pistil_base, pistil_tip, (255, 255, 255), 2)

    info = direction_from_points(pistil_base, pistil_tip)
    if info is None:
        cv2.putText(out, "Khong tinh duoc vector huong", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return out, None

    L = int(max(70, vector_scale * info["norm"]))
    end_pt = (
        int(round(pistil_base[0] + L * info["ux"])),
        int(round(pistil_base[1] + L * info["uy"])),
    )

    cv2.arrowedLine(out, pistil_base, end_pt, (255, 255, 255), 5, tipLength=0.18)

    cv2.putText(out, f"P1 = {pistil_base}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"P2 = {pistil_tip}", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"dx = {info['dx']:.2f}, dy = {info['dy']:.2f}", (15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"theta_axis = {info['theta_axis_deg']:.2f} deg", (15, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Doan xanh: P1 -> tam hoa | Mui ten trang: huong nhị P1 -> P2", (15, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    return out, info


def draw_lab_channels_with_points(L_channel, A_channel, B_channel, p1, p2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    channels = [("L channel", L_channel), ("A channel", A_channel), ("B channel", B_channel)]

    for ax, (title, ch) in zip(axes, channels):
        ax.imshow(ch, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

        if p1 is not None:
            x1, y1 = p1
            if 0 <= x1 < ch.shape[1] and 0 <= y1 < ch.shape[0]:
                v1 = int(ch[y1, x1])
                ax.plot(x1, y1, "ro", markersize=7)
                ax.text(
                    x1 + 5,
                    y1 - 5,
                    f"P1={v1}",
                    color="red",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

        if p2 is not None:
            x2, y2 = p2
            if 0 <= x2 < ch.shape[1] and 0 <= y2 < ch.shape[0]:
                v2 = int(ch[y2, x2])
                ax.plot(x2, y2, "bo", markersize=7)
                ax.text(
                    x2 + 5,
                    y2 + 12,
                    f"P2={v2}",
                    color="blue",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

    plt.tight_layout()
    return fig


def draw_pistil_selection_debug_image(
    img_bgr,
    flower_cnt,
    pistil_cnt,
    flower_minor_info,
    pistil_axis_info,
    pistil_base,
    pistil_tip,
    flower_center,
    vector_scale=12.0,
):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    if flower_cnt is not None:
        cv2.drawContours(out, [flower_cnt], -1, (255, 0, 0), 2)

    if pistil_cnt is not None:
        cv2.drawContours(out, [pistil_cnt], -1, (0, 0, 255), 2)

    if flower_minor_info:
        ellipse = flower_minor_info.get("ellipse", None)
        minor_dir = flower_minor_info.get("minor_dir", None)

        if ellipse is not None:
            cv2.ellipse(out, ellipse, (0, 255, 0), 1)

        if flower_center is not None:
            cv2.circle(out, flower_center, 6, (0, 255, 0), -1)

        if flower_center is not None and minor_dir is not None:
            L = int(np.hypot(h, w))
            p1 = (
                int(round(flower_center[0] - L * minor_dir[0])),
                int(round(flower_center[1] - L * minor_dir[1])),
            )
            p2 = (
                int(round(flower_center[0] + L * minor_dir[0])),
                int(round(flower_center[1] + L * minor_dir[1])),
            )
            cv2.line(out, p1, p2, (255, 0, 255), 2)

    ellipse = pistil_axis_info.get("ellipse", None)
    if ellipse is not None:
        cv2.ellipse(out, ellipse, (255, 255, 0), 2)

    circle = pistil_axis_info.get("circle", None)
    if circle is not None:
        (ccx, ccy), cr = circle
        cv2.circle(out, (int(round(ccx)), int(round(ccy))), int(round(cr)), (0, 165, 255), 2)

    cand_a = pistil_axis_info.get("candidate_a", None)
    cand_b = pistil_axis_info.get("candidate_b", None)
    d1 = pistil_axis_info.get("dist_a_to_minor_axis", None)
    d2 = pistil_axis_info.get("dist_b_to_minor_axis", None)
    l1 = pistil_axis_info.get("L_candidate_a", None)
    l2 = pistil_axis_info.get("L_candidate_b", None)
    axis_point = pistil_axis_info.get("minor_axis_point", None)
    axis_dir = pistil_axis_info.get("minor_axis_dir", None)

    for pt, color, label, dist_val, l_val, dy_txt in [
        (cand_a, (0, 0, 255), "A", d1, l1, -10),
        (cand_b, (255, 0, 0), "B", d2, l2, 18),
    ]:
        if pt is None:
            continue

        cv2.circle(out, pt, 7, color, -1)

        proj = project_point_to_line(pt, axis_point, axis_dir)
        if proj is not None:
            proj_int = tuple(np.round(proj).astype(int))
            cv2.line(out, pt, proj_int, color, 2)
            cv2.circle(out, proj_int, 4, color, -1)

        text = label
        if dist_val is not None:
            text += f" d={dist_val:.2f}"
        if l_val is not None:
            text += f" L={l_val:.1f}"

        cv2.putText(
            out,
            text,
            (pt[0] + 8, pt[1] + dy_txt),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    pistil_mode = pistil_axis_info.get("mode", "")
    if pistil_mode == "circle_like":
        center_pt = pistil_axis_info.get("center", pistil_base)
        if center_pt is not None:
            cv2.circle(out, center_pt, 6, (255, 255, 0), -1)
            if flower_center is not None:
                cv2.line(out, center_pt, flower_center, (0, 255, 0), 2)
            cv2.putText(
                out,
                "Circle-like -> chi xet tam, khong dung truc chinh",
                (15, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        if pistil_base is not None:
            cv2.circle(out, pistil_base, 6, (255, 255, 0), -1)
        if pistil_tip is not None:
            cv2.circle(out, pistil_tip, 6, (0, 255, 255), -1)

        if pistil_base is not None and pistil_tip is not None:
            if flower_center is not None:
                cv2.line(out, pistil_base, flower_center, (0, 255, 0), 2)

            cv2.line(out, pistil_base, pistil_tip, (255, 255, 255), 2)

            dir_info = direction_from_points(pistil_base, pistil_tip)
            if dir_info is not None:
                L_vec = int(max(70, vector_scale * dir_info["norm"]))
                end_pt = (
                    int(round(pistil_base[0] + L_vec * dir_info["ux"])),
                    int(round(pistil_base[1] + L_vec * dir_info["uy"])),
                )
                cv2.arrowedLine(out, pistil_base, end_pt, (255, 255, 255), 4, tipLength=0.18)
                cv2.putText(
                    out,
                    f"theta_axis = {dir_info['theta_axis_deg']:.2f} deg",
                    (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    dist_gap = pistil_axis_info.get("dist_gap", None)
    axis_dist_delta = pistil_axis_info.get("axis_dist_delta", None)
    selected_by = pistil_axis_info.get("selected_by", "unknown")
    minor_axis_available = pistil_axis_info.get("minor_axis_available", False)
    circle_iou = pistil_axis_info.get("circle_iou", None)
    ellipse_iou = pistil_axis_info.get("ellipse_iou", None)
    shape_selected = pistil_axis_info.get("shape_selected", "unknown")

    cv2.putText(out, f"shape_selected = {shape_selected}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if circle_iou is not None and ellipse_iou is not None:
        cv2.putText(out, f"IoU circle = {circle_iou:.3f} | ellipse = {ellipse_iou:.3f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"minor axis available = {minor_axis_available}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"selected_by = {selected_by}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    if dist_gap is not None and axis_dist_delta is not None:
        cv2.putText(out, f"dist_gap = {dist_gap:.2f} | delta = {axis_dist_delta:.2f}", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def draw_pistil_mask_on_lab(L_channel, pistil_mask):
    L_rgb = cv2.cvtColor(L_channel, cv2.COLOR_GRAY2BGR)
    out = L_rgb.copy()
    out[pistil_mask > 0] = [0, 0, 255]
    return out


# =========================
# MAIN
# =========================
def main():
    img_path = os.path.join(".", ROOT_PATH, IMG_NAME)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh {img_path}")

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_channel, A_channel, B_channel = cv2.split(img_lab)

    t0 = time.perf_counter()

    flower_mask = flower_mask_hsv(img, min_component_area=FLOWER_MIN_COMPONENT_AREA)
    flower_only = apply_mask_to_bgr(img, flower_mask)
    flower_cnt = largest_contour_from_mask(flower_mask)
    flower_center, flower_geom_info = flower_reference_center_from_outer_contour(
        flower_cnt,
        mode=FLOWER_CENTER_MODE,
    )
    flower_minor_info = flower_minor_axis_from_contour(flower_cnt)

    pistil_mask, pistil_score_vis = pistil_mask_from_flower_only_rgb(
        flower_only,
        flower_mask,
        center_radius_ratio=CENTER_RADIUS_RATIO,
        r_min=100,
        r_max=200,
        g_min=100,
        g_max=210,
    )
    pistil_cnt = largest_contour_from_mask(pistil_mask)

    pistil_base, pistil_tip, pistil_axis_info = pistil_base_tip_by_iou(
        pistil_cnt,
        flower_center,
        flower_minor_info=flower_minor_info,
        L_channel=L_channel,
        axis_dist_delta=MINOR_AXIS_DIST_DELTA,
        lab_patch_radius=LAB_PATCH_RADIUS,
        image_shape=img.shape[:2],
    )

    candidate_a = pistil_axis_info.get("candidate_a", None)
    candidate_b = pistil_axis_info.get("candidate_b", None)
    La = pistil_axis_info.get("L_candidate_a", None)
    Lb = pistil_axis_info.get("L_candidate_b", None)
    dist_a = pistil_axis_info.get("dist_a_to_minor_axis", None)
    dist_b = pistil_axis_info.get("dist_b_to_minor_axis", None)
    dist_gap = pistil_axis_info.get("dist_gap", None)
    selected_by = pistil_axis_info.get("selected_by", None)
    minor_axis_available = pistil_axis_info.get("minor_axis_available", None)
    circle_iou = pistil_axis_info.get("circle_iou", None)
    ellipse_iou = pistil_axis_info.get("ellipse_iou", None)
    shape_selected = pistil_axis_info.get("shape_selected", None)

    txt_path = os.path.join(".", OUTPUT_DIR, f"{os.path.splitext(IMG_NAME)[0]}_geometry.txt")
    save_geometry_to_txt(
        txt_path=txt_path,
        img_name=IMG_NAME,
        flower_mask=flower_mask,
        flower_cnt=flower_cnt,
        flower_geom_info=flower_geom_info,
        flower_minor_info=flower_minor_info,
        pistil_mask=pistil_mask,
        pistil_cnt=pistil_cnt,
        pistil_axis_info=pistil_axis_info,
    )

    t1 = time.perf_counter()

    print(f"latency: {t1 - t0:.6f} s")
    print("flower_center  =", flower_center)
    print("pistil_point_1 =", pistil_base)
    print("pistil_point_2 =", pistil_tip)
    print("candidate_a    =", candidate_a, "L_mean =", La, "dist_to_minor_axis =", dist_a)
    print("candidate_b    =", candidate_b, "L_mean =", Lb, "dist_to_minor_axis =", dist_b)
    print("dist_gap       =", dist_gap)
    print("minor axis ok  =", minor_axis_available)
    print("selected_by    =", selected_by)
    print("shape_selected =", shape_selected)
    print("circle_iou     =", circle_iou)
    print("ellipse_iou    =", ellipse_iou)
    print("txt_path       =", txt_path)

    if flower_cnt is not None:
        print(f"flower contour area = {cv2.contourArea(flower_cnt):.2f}")
    else:
        print("Khong tim thay flower contour.")

    if pistil_cnt is not None:
        print(f"pistil contour area = {cv2.contourArea(pistil_cnt):.2f}")
    else:
        print("Khong tim thay pistil contour.")

    img_detect = draw_detection_image(
        img,
        flower_cnt,
        pistil_cnt,
        flower_center,
        pistil_base,
        pistil_tip,
        flower_geom_info,
        flower_minor_info,
        pistil_axis_info,
    )

    img_vector, dir_info = draw_direction_vector_image(
        img,
        flower_cnt,
        pistil_cnt,
        flower_center,
        pistil_base,
        pistil_tip,
        pistil_axis_info,
        vector_scale=VECTOR_SCALE,
    )

    img_select_debug = draw_pistil_selection_debug_image(
        img,
        flower_cnt,
        pistil_cnt,
        flower_minor_info,
        pistil_axis_info,
        pistil_base,
        pistil_tip,
        flower_center,
        vector_scale=VECTOR_SCALE,
    )

    img_pistil_lab_mask = draw_pistil_mask_on_lab(L_channel, pistil_mask)

    if dir_info is not None:
        print(f"pistil dx = {dir_info['dx']:.3f}, dy = {dir_info['dy']:.3f}")
        print(f"pistil theta_axis_deg = {dir_info['theta_axis_deg']:.3f}")
    else:
        print("Khong tinh / khong ve vector huong.")

    plt.figure(figsize=(24, 12))

    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 0 - Anh goc")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 1 - Vien/mask, ellipse/rect, P1-P2")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(flower_mask, cmap="gray")
    plt.title("Hinh 2 - Flower mask")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(pistil_mask, cmap="gray")
    plt.title("Hinh 3 - Pistil mask")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(img_vector, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 4 - Huong nhị P1 -> P2")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(pistil_score_vis, cmap="gray")
    plt.title("Hinh 5 - Pistil score")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(img_select_debug, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 6 - IoU ellipse/rect + chon P1/P2")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(img_pistil_lab_mask, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 7 - Mask nhi tren kenh L")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    draw_lab_channels_with_points(
        L_channel,
        A_channel,
        B_channel,
        candidate_a,
        candidate_b,
    )
    plt.show()


if __name__ == "__main__":
    main()
