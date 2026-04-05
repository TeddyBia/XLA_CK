
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


# =========================
# CONFIG
# =========================
IMG_NAME = "1.png"
ROOT_PATH = "img"
APS_TH = 1.4

# "circle" bám sát ý minimum circumscribed circle
# "ellipse" nếu muốn thử fit ellipse
FLOWER_CENTER_MODE = "ellipse"

# tham số tách nhụy
PISTIL_PERCENTILE = 80
CENTER_RADIUS_RATIO = 0.12

# scale hiển thị véc tơ
VECTOR_SCALE = 12.0


# =========================
# BASIC HELPERS
# =========================
def to_gray(img_bgr, crop_border=0):
    work = img_bgr.copy()
    if (
        crop_border > 0
        and work.shape[0] > 2 * crop_border
        and work.shape[1] > 2 * crop_border
    ):
        work = work[crop_border:-crop_border, crop_border:-crop_border]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    return gray


def apply_mask_to_bgr(img_bgr, mask):
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def normalize_to_uint8(x):
    x = x.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)

    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)

    y = (x - x_min) / (x_max - x_min)
    y = np.clip(255.0 * y, 0, 255).astype(np.uint8)
    return y


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.where(labels == largest_id, 255, 0).astype(np.uint8)
    return out


def largest_contour_from_mask(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
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


# =========================
# FLOWER MASK
# =========================
def flower_mask_hsv(img_bgr, keep_largest=True):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ngưỡng vàng của hoa
    lower = np.array([18, 80, 30], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # hậu xử lý
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    if keep_largest:
        mask = keep_largest_component(mask)

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

    out = np.where(labels == best_id, 255, 0).astype(np.uint8)
    return out


def pistil_mask_from_flower_only(
    flower_only_bgr,
    flower_mask,
    pistil_percentile=80,
    center_radius_ratio=0.22
):
    # co nhẹ vùng hoa
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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

    # score xanh so với vàng
    score = (g - r) + 0.01 * (g - b)

    valid = (inner_flower > 0) & center_gate
    vals = score[valid]

    if vals.size == 0:
        return np.zeros_like(flower_mask), np.zeros_like(flower_mask)

    thr = np.percentile(vals, pistil_percentile)

    pistil = np.zeros_like(flower_mask)
    pistil[(score >= thr) & valid] = 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    pistil = cv2.morphologyEx(pistil, cv2.MORPH_OPEN, kernel_small, iterations=1)
    pistil = cv2.morphologyEx(pistil, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    pistil = keep_component_nearest_center(pistil, cx, cy, min_area=3)

    score_vis = normalize_to_uint8(score)
    return pistil, score_vis


# =========================
# FLOWER CENTER: petal-plane reference
# =========================
def flower_reference_center_from_outer_contour(cnt, mode="circle"):
    """
    Tâm hoa tham chiếu:
    - circle: tâm minimum enclosing circle của contour ngoài
    - ellipse: tâm fitEllipse nếu đủ điểm
    """
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


# =========================
# PISTIL BASE POINT
# =========================
def pistil_axis_endpoints_from_contour(cnt):
    """
    Fit 1 đường thẳng qua contour nhụy, rồi lấy 2 đầu mút theo phương trục.
    """
    if cnt is None:
        return None, None, {}

    pts = cnt.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 2:
        p = tuple(np.round(pts[0]).astype(int))
        return p, p, {}

    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    d = np.array([vx, vy], dtype=np.float32)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        c = contour_centroid_from_contour(cnt)
        return c, c, {}

    d = d / d_norm
    p0 = np.array([x0, y0], dtype=np.float32)

    t = (pts - p0) @ d
    t_min = float(np.min(t))
    t_max = float(np.max(t))

    p_a = p0 + t_min * d
    p_b = p0 + t_max * d

    p_a_int = tuple(np.round(p_a).astype(int))
    p_b_int = tuple(np.round(p_b).astype(int))

    info = {
        "fit_point": (float(x0), float(y0)),
        "fit_dir": (float(d[0]), float(d[1])),
        "p_a": p_a_int,
        "p_b": p_b_int,
    }
    return p_a_int, p_b_int, info

def pistil_rotated_rect_from_contour(cnt):
    """
    Fit nhụy theo hình chữ nhật nghiêng nhỏ nhất.
    Trả về:
    - rect: ((cx, cy), (w, h), angle)
    - box: 4 đỉnh của rotated rectangle
    """
    if cnt is None or len(cnt) < 3:
        return None, None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(int)
    return rect, box

def pistil_base_tip_hybrid(cnt, flower_center, aspect_thr=1.6):
    """
    Nếu nhụy thuôn dài -> dùng rotated rect để lấy base-tip.
    Nếu nhụy gần tròn -> chỉ lấy tâm, base = tip = tâm nhụy.
    """
    if cnt is None:
        return None, None, {}

    rect, box = pistil_rotated_rect_from_contour(cnt)
    if rect is None:
        c = contour_centroid_from_contour(cnt)
        return c, c, {"mode": "centroid"}

    (cx, cy), (w, h), angle = rect
    long_side = max(w, h)
    short_side = max(min(w, h), 1e-6)
    aspect_ratio = long_side / short_side

    # gần chính diện / gần tròn
    if aspect_ratio < aspect_thr:
        c = (int(round(cx)), int(round(cy)))
        info = {
            "mode": "frontal_circle_like",
            "rect": rect,
            "box": box,
            "center": c,
            "aspect_ratio": float(aspect_ratio),
        }
        return c, c, info

    # thuôn dài
    base_pt, tip_pt, info = pistil_base_and_tip_from_rotated_rect(cnt, flower_center)
    info["mode"] = "elongated_rect"
    info["aspect_ratio"] = float(aspect_ratio)
    return base_pt, tip_pt, info

def pistil_circle_from_contour(cnt):
    if cnt is None:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return (int(round(cx)), int(round(cy))), int(round(r))

def pistil_base_and_tip_from_rotated_rect(cnt, flower_center):
    """
    Dùng rotated rectangle của nhụy để lấy:
    - base_pt: đầu của trục dài gần tâm hoa hơn
    - tip_pt : đầu còn lại
    """
    if cnt is None or flower_center is None or len(cnt) < 3:
        return None, None, {}

    rect, box = pistil_rotated_rect_from_contour(cnt)
    if rect is None:
        return None, None, {}

    (cx, cy), (w, h), angle = rect
    center = np.array([cx, cy], dtype=np.float32)

    box_f = box.astype(np.float32)

    # 4 cạnh của hình chữ nhật
    edges = []
    for i in range(4):
        p1 = box_f[i]
        p2 = box_f[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        mid_pt = 0.5 * (p1 + p2)
        edges.append({
            "p1": p1,
            "p2": p2,
            "vec": edge_vec,
            "len": edge_len,
            "mid": mid_pt
        })

    # minAreaRect có 2 cạnh dài, 2 cạnh ngắn
    # chọn 1 cạnh dài để xác định phương trục dài
    long_edge = max(edges, key=lambda e: e["len"])
    d = long_edge["vec"]
    d_norm = np.linalg.norm(d)

    if d_norm < 1e-12:
        c = (int(round(cx)), int(round(cy)))
        return c, c, {"rect": rect, "box": box}

    d = d / d_norm

    # lấy nửa chiều dài theo trục dài
    half_long = max(w, h) / 2.0

    p_a = center - half_long * d
    p_b = center + half_long * d

    p_a_int = tuple(np.round(p_a).astype(int))
    p_b_int = tuple(np.round(p_b).astype(int))

    fc = np.array(flower_center, dtype=np.float32)
    da = np.linalg.norm(p_a - fc)
    db = np.linalg.norm(p_b - fc)

    if da <= db:
        base_pt = p_a_int
        tip_pt = p_b_int
    else:
        base_pt = p_b_int
        tip_pt = p_a_int

    info = {
        "rect": rect,
        "box": box,
        "center": (float(cx), float(cy)),
        "size": (float(w), float(h)),
        "angle": float(angle),
        "base_pt": base_pt,
        "tip_pt": tip_pt,
    }
    return base_pt, tip_pt, info

def pistil_base_and_tip_from_contour(cnt, flower_center):
    """
    Chọn:
    - gốc nhụy = đầu của trục nhụy gần tâm hoa nhất
    - đầu nhụy = đầu còn lại
    """
    if cnt is None or flower_center is None:
        return None, None, {}

    p_a, p_b, info = pistil_axis_endpoints_from_contour(cnt)
    if p_a is None or p_b is None:
        return None, None, info

    fc = np.array(flower_center, dtype=np.float32)
    pa = np.array(p_a, dtype=np.float32)
    pb = np.array(p_b, dtype=np.float32)

    da = np.linalg.norm(pa - fc)
    db = np.linalg.norm(pb - fc)

    if da <= db:
        base_pt = p_a
        tip_pt = p_b
    else:
        base_pt = p_b
        tip_pt = p_a

    info["base_pt"] = base_pt
    info["tip_pt"] = tip_pt
    return base_pt, tip_pt, info


# =========================
# DIRECTION
# =========================
def flower_direction_from_points(p_from, p_to):
    """
    Véc tơ hướng từ p_from -> p_to
    """
    if p_from is None or p_to is None:
        return None

    x1, y1 = p_from
    x2, y2 = p_to

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    norm = np.hypot(dx, dy)
    if norm < 1e-9:
        return None

    ux = dx / norm
    uy = dy / norm

    theta_math_deg = np.degrees(np.arctan2(-dy, dx))
    theta_axis_deg = theta_math_deg % 180.0

    return {
        "dx": dx,
        "dy": dy,
        "norm": norm,
        "ux": ux,
        "uy": uy,
        "theta_math_deg": float(theta_math_deg),
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
    pistil_axis_info
):
    """
    Ảnh 1:
    - contour hoa
    - contour nhụy
    - tâm hoa tham chiếu
    - gốc nhụy calib
    - đầu nhụy
    - hình fit của nhụy sẽ phụ thuộc mode:
        + elongated_rect      -> vẽ rotated rectangle
        + frontal_circle_like -> vẽ circle
    """
    out = img_bgr.copy()

    # contour hoa
    if flower_cnt is not None:
        cv2.drawContours(out, [flower_cnt], -1, (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(flower_cnt)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # contour nhụy
    if pistil_cnt is not None:
        cv2.drawContours(out, [pistil_cnt], -1, (0, 0, 255), 2)

        pistil_mode = pistil_axis_info.get("mode", "")

        if pistil_mode == "elongated_rect":
            box = pistil_axis_info.get("box", None)
            if box is not None:
                cv2.drawContours(out, [np.asarray(box, dtype=np.int32)], 0, (0, 165, 255), 2)

        elif pistil_mode == "frontal_circle_like":
            circle_data = pistil_circle_from_contour(pistil_cnt)
            if circle_data is not None:
                c, r = circle_data
                cv2.circle(out, c, r, (0, 165, 255), 2)

        else:
            # fallback: nếu chưa có mode rõ ràng thì vẽ theo contour thôi
            pass

    # tâm hoa kiểu circle/ellipse
    if flower_center is not None:
        cv2.circle(out, flower_center, 6, (0, 255, 0), -1)
        cv2.putText(
            out, f"Flower C {flower_center}", (flower_center[0] + 8, flower_center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA
        )

    if flower_geom_info.get("mode") == "circle":
        (cx, cy), r = flower_geom_info["circle"]
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 1)

    elif flower_geom_info.get("mode") == "ellipse":
        ellipse = flower_geom_info["ellipse"]
        cv2.ellipse(out, ellipse, (0, 255, 0), 1)

    # gốc nhụy calib
    if pistil_base is not None:
        cv2.circle(out, pistil_base, 6, (255, 255, 0), -1)
        cv2.putText(
            out, f"Pistil calib {pistil_base}", (pistil_base[0] + 8, pistil_base[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 0), 2, cv2.LINE_AA
        )

    # đầu nhụy
    if pistil_tip is not None:
        cv2.circle(out, pistil_tip, 6, (0, 255, 255), -1)
        cv2.putText(
            out, f"Pistil tip {pistil_tip}", (pistil_tip[0] + 8, pistil_tip[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA
        )

    # trục nhụy: chỉ vẽ khi nhụy thật sự thuôn dài
    if pistil_axis_info.get("mode") == "elongated_rect":
        if pistil_base is not None and pistil_tip is not None:
            cv2.line(out, pistil_base, pistil_tip, (255, 255, 255), 2)

    return out

def draw_direction_vector_image(
    img_bgr,
    flower_cnt,
    pistil_cnt,
    flower_center,
    pistil_base,
    pistil_tip,
    vector_scale=12.0
):
    """
    Ảnh 4:
    - tâm hoa = tâm circle/ellipse của cánh
    - điểm nhụy calib = gốc nhụy gần tâm hoa nhất
    - véc tơ hướng từ tâm nhụy calib -> tâm hoa
    """
    out = img_bgr.copy()

    if flower_cnt is not None:
        cv2.drawContours(out, [flower_cnt], -1, (255, 0, 0), 2)

    if pistil_cnt is not None:
        cv2.drawContours(out, [pistil_cnt], -1, (0, 0, 255), 2)

    if flower_center is not None:
        cv2.circle(out, flower_center, 6, (0, 255, 0), -1)

    if pistil_base is not None:
        cv2.circle(out, pistil_base, 6, (255, 255, 0), -1)

    if pistil_tip is not None:
        cv2.circle(out, pistil_tip, 6, (0, 255, 255), -1)
        cv2.line(out, pistil_base, pistil_tip, (255, 255, 255), 2)

    info = flower_direction_from_points(pistil_base, flower_center)
    if info is None:
        return out, None

    ux = info["ux"]
    uy = info["uy"]

    L = int(max(70, vector_scale * info["norm"]))
    end_pt = (
        int(round(pistil_base[0] + L * ux)),
        int(round(pistil_base[1] + L * uy))
    )

    cv2.arrowedLine(
        out,
        pistil_base,
        end_pt,
        (0, 0, 255),
        5,
        tipLength=0.18
    )

    cv2.putText(
        out, f"Flower C = {flower_center}", (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        out, f"Pistil calib = {pistil_base}", (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        out, f"dx = {info['dx']:.2f}, dy = {info['dy']:.2f}", (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        out, f"theta_axis = {info['theta_axis_deg']:.2f} deg", (15, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA
    )

    return out, info


# =========================
# MAIN
# =========================
def main():
    img_path = os.path.join(".", ROOT_PATH, IMG_NAME)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh {img_path}")

    gray = to_gray(img)

    t0 = time.perf_counter()

    # mask hoa
    flower_mask = flower_mask_hsv(img, keep_largest=True)
    flower_only = apply_mask_to_bgr(img, flower_mask)

    # contour ngoài của hoa
    flower_cnt = largest_contour_from_mask(flower_mask)

    # mask nhụy
    pistil_mask, pistil_score_vis = pistil_mask_from_flower_only(
        flower_only,
        flower_mask,
        pistil_percentile=PISTIL_PERCENTILE,
        center_radius_ratio=CENTER_RADIUS_RATIO
    )

    # contour nhụy
    pistil_cnt = largest_contour_from_mask(pistil_mask)

    # tâm hoa kiểu cũ: từ cánh
    flower_center, flower_geom_info = flower_reference_center_from_outer_contour(
        flower_cnt,
        mode=FLOWER_CENTER_MODE
    )

    # calib nhụy kiểu cũ: fitLine -> chọn endpoint gần tâm hoa nhất
    pistil_base, pistil_tip, pistil_axis_info = pistil_base_tip_hybrid(
        pistil_cnt,
        flower_center,
        aspect_thr=APS_TH
    )
    t1 = time.perf_counter()

    print(f"latency: {t1 - t0:.6f} s")
    print("flower_center =", flower_center)
    print("pistil_calib  =", pistil_base)
    print("pistil_tip    =", pistil_tip)

    if flower_cnt is not None:
        print(f"flower contour area = {cv2.contourArea(flower_cnt):.2f}")
    else:
        print("Khong tim thay flower contour.")

    if pistil_cnt is not None:
        print(f"pistil contour area = {cv2.contourArea(pistil_cnt):.2f}")
    else:
        print("Khong tim thay pistil contour.")

    # Hình 1
    img_detect = draw_detection_image(
        img,
        flower_cnt,
        pistil_cnt,
        flower_center,
        pistil_base,
        pistil_tip,
        flower_geom_info,
        pistil_axis_info
    )

    # Hình 4
    img_vector, dir_info = draw_direction_vector_image(
        img,
        flower_cnt,
        pistil_cnt,
        flower_center,
        pistil_base,
        pistil_tip,
        vector_scale=VECTOR_SCALE
    )

    if dir_info is not None:
        print(f"dx = {dir_info['dx']:.3f}, dy = {dir_info['dy']:.3f}")
        print(f"theta_axis_deg = {dir_info['theta_axis_deg']:.3f}")

    # ===== chỉ hiển thị đúng 4 hình =====
    # ===== hiển thị 5 hình + ảnh gốc =====
    plt.figure(figsize=(20, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 0 - Anh goc")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 1 - Vien hoa/nhuy va tam tren anh goc")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(flower_mask, cmap="gray")
    plt.title("Hinh 2 - Flower mask")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(pistil_mask, cmap="gray")
    plt.title("Hinh 3 - Pistil mask")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img_vector, cv2.COLOR_BGR2RGB))
    plt.title("Hinh 4 - Vector tu pistil calib den tam hoa")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(pistil_score_vis, cmap="gray")
    plt.title("Hinh 5 - Pistil score")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
