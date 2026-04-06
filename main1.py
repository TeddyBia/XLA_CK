import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


# =========================================================
# THAM SO CHUNG
# =========================================================
DRAW_THICKNESS = 1
FLOWER_MIN_COMPONENT_AREA = 100
PISTIL_MIN_COMPONENT_AREA = 0

FLOWER_SEGMENT_METHOD = "hsv"   # "hsv" | "otsu_gray" | "rgb"
AUTO_SAVE_IMAGES = True


def apply_mask_to_bgr(img_bgr, mask):
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_image(save_path, image):
    """
    Lưu ảnh BGR hoặc grayscale.
    """
    ensure_parent_dir(save_path)
    ok = cv2.imwrite(save_path, image)
    if not ok:
        raise IOError(f"Khong luu duoc anh: {save_path}")


def save_mask_image(
    mask,
    save_path,
    invert=False,
    as_bgr=False,
    scale=1.0
):
    """
    Hàm riêng để lưu mask theo ý muốn.
    Tham số:
    - mask     : mask đầu vào
    - save_path: đường dẫn lưu
    - invert   : đảo mask trước khi lưu hay không
    - as_bgr   : lưu mask dưới dạng 3 kênh hay không
    - scale    : scale ảnh khi lưu (ví dụ 2.0 để phóng to)
    """
    if mask is None:
        raise ValueError("mask is None")

    out = mask.copy()

    if out.dtype != np.uint8:
        out = out.astype(np.uint8)

    if out.max() <= 1:
        out = out * 255

    out = np.where(out > 0, 255, 0).astype(np.uint8)

    if invert:
        out = cv2.bitwise_not(out)

    if scale is not None and abs(scale - 1.0) > 1e-9:
        h, w = out.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if as_bgr:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    save_image(save_path, out)


def save_output_images(output_dir, original_img, result_img, flower_mask, pistil_mask):
    """
    Lưu 4 ảnh chuẩn:
    - ảnh gốc
    - ảnh detect
    - mask hoa
    - mask nhụy
    """
    os.makedirs(output_dir, exist_ok=True)

    save_image(os.path.join(output_dir, "01_original.png"), original_img)
    save_image(os.path.join(output_dir, "02_detected.png"), result_img)
    save_mask_image(flower_mask, os.path.join(output_dir, "03_flower_mask.png"))
    save_mask_image(pistil_mask, os.path.join(output_dir, "04_pistil_mask.png"))


def remove_small_components(mask, min_area=300):
    """
    Loại bỏ các cụm liên thông nhỏ hơn min_area.
    Giữ lại tất cả vùng đủ lớn.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    out = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label_id] = 255

    return out


def split_connected_components(mask, min_area=1):
    """
    Tách từng cụm liên thông thành list mask riêng.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            comp = np.zeros_like(mask)
            comp[labels == label_id] = 255
            components.append(comp)

    return components


def flower_mask_hsv(img_bgr, remove_small=True, min_component_area=300):
    """
    Tách mask hoa bằng HSV.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 0, 0], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    if remove_small:
        mask = remove_small_components(mask, min_area=min_component_area)

    return mask


def flower_mask_otsu_gray(
    img_bgr,
    blur_ksize=5,
    invert=False,
    remove_small=True,
    min_component_area=300
):
    """
    Phân ngưỡng Otsu theo ảnh grayscale.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if blur_ksize is not None and blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, thresh_type | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    if remove_small:
        mask = remove_small_components(mask, min_area=min_component_area)

    return mask


def flower_mask_rgb(
    img_bgr,
    r_range=(120, 255),
    g_range=(80, 255),
    b_range=(0, 200),
    remove_small=True,
    min_component_area=300
):
    """
    Phân ngưỡng theo RGB.
    Bạn có thể chỉnh các khoảng r_range, g_range, b_range theo ý muốn.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    mask = (
        (r >= r_range[0]) & (r <= r_range[1]) &
        (g >= g_range[0]) & (g <= g_range[1]) &
        (b >= b_range[0]) & (b <= b_range[1])
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    if remove_small:
        mask = remove_small_components(mask, min_area=min_component_area)

    return mask


def get_flower_mask(img_bgr, method="hsv", min_component_area=300):
    """
    Bộ chọn phương pháp phân ngưỡng hoa.
    """
    method = method.lower().strip()

    if method == "hsv":
        return flower_mask_hsv(
            img_bgr,
            remove_small=True,
            min_component_area=min_component_area
        )

    if method == "otsu_gray":
        return flower_mask_otsu_gray(
            img_bgr,
            invert=False,
            remove_small=True,
            min_component_area=min_component_area
        )

    if method == "rgb":
        return flower_mask_rgb(
            img_bgr,
            r_range=(120, 255),
            g_range=(80, 255),
            b_range=(0, 200),
            remove_small=True,
            min_component_area=min_component_area
        )

    raise ValueError(f"Unsupported FLOWER_SEGMENT_METHOD: {method}")


def get_largest_external_contour(mask):
    """
    Lấy contour ngoài lớn nhất của một mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def contour_to_filled_mask(shape_hw, cnt):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if cnt is not None:
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    return mask


def fit_ellipse_from_contour(cnt):
    """
    Fit ellipse từ contour.
    """
    if cnt is None:
        return None, None

    if len(cnt) < 5:
        return None, None

    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), _, _ = ellipse
    center = (int(round(cx)), int(round(cy)))

    return ellipse, center


def compute_direction_info(center_flower, center_pistil):
    """
    Hướng véc tơ: từ tâm nhị/nhụy -> tâm hoa
    """
    if center_flower is None or center_pistil is None:
        return None

    cx_f, cy_f = center_flower
    cx_p, cy_p = center_pistil

    dx = float(cx_f - cx_p)
    dy = float(cy_f - cy_p)

    norm = np.hypot(dx, dy)
    if norm < 1e-9:
        return None

    ux = dx / norm
    uy = dy / norm

    theta_img_deg = float(np.degrees(np.arctan2(dy, dx)))
    theta_math_deg = float(np.degrees(np.arctan2(-dy, dx)))
    theta_axis_deg = theta_math_deg % 180.0

    return {
        "dx": dx,
        "dy": dy,
        "norm": norm,
        "ux": ux,
        "uy": uy,
        "theta_img_deg": theta_img_deg,
        "theta_math_deg": theta_math_deg,
        "theta_axis_deg": theta_axis_deg,
    }


def find_pistil_masks_by_inverting_flower(flower_component_mask, min_component_area=20):
    outer_cnt = get_largest_external_contour(flower_component_mask)
    if outer_cnt is None:
        return [], None, None

    filled_outer = contour_to_filled_mask(flower_component_mask.shape, outer_cnt)

    inv_flower = cv2.bitwise_not(flower_component_mask)
    pistil_candidate = cv2.bitwise_and(inv_flower, filled_outer)

    largest_pistil_cnt = get_largest_external_contour(pistil_candidate)

    if largest_pistil_cnt is None:
        pistil_components = []
    else:
        if cv2.contourArea(largest_pistil_cnt) < float(min_component_area):
            pistil_components = []
        else:
            largest_pistil_mask = contour_to_filled_mask(
                flower_component_mask.shape,
                largest_pistil_cnt
            )
            pistil_components = [largest_pistil_mask]

    return pistil_components, outer_cnt, filled_outer


def analyze_all_flowers_and_pistils(
    img_bgr,
    flower_min_component_area=300,
    pistil_min_component_area=20,
    flower_segment_method="hsv"
):
    """
    Phân tích toàn bộ:
    - tất cả flower masks
    - tất cả mask nhụy tìm được từ từng flower bằng INV
    """
    flower_mask_all = get_flower_mask(
        img_bgr,
        method=flower_segment_method,
        min_component_area=flower_min_component_area
    )

    flower_components = split_connected_components(
        flower_mask_all,
        min_area=flower_min_component_area
    )

    all_results = []
    pistil_mask_all = np.zeros_like(flower_mask_all)

    for flower_idx, flower_comp_mask in enumerate(flower_components, start=1):
        pistil_components, outer_cnt, filled_outer = find_pistil_masks_by_inverting_flower(
            flower_comp_mask,
            min_component_area=pistil_min_component_area
        )

        flower_ellipse, center_flower = fit_ellipse_from_contour(outer_cnt)

        flower_info = {
            "flower_idx": flower_idx,
            "flower_mask": flower_comp_mask,
            "flower_contour": outer_cnt,
            "flower_filled_mask": filled_outer,
            "flower_ellipse": flower_ellipse,
            "center_flower": center_flower,
            "flower_area": float(cv2.contourArea(outer_cnt)) if outer_cnt is not None else 0.0,
            "pistils": []
        }

        for pistil_idx, pistil_comp_mask in enumerate(pistil_components, start=1):
            pistil_mask_all = cv2.bitwise_or(pistil_mask_all, pistil_comp_mask)

            pistil_cnt = get_largest_external_contour(pistil_comp_mask)
            pistil_ellipse, center_pistil = fit_ellipse_from_contour(pistil_cnt)
            dir_info = compute_direction_info(center_flower, center_pistil)

            pistil_info = {
                "pistil_idx": pistil_idx,
                "pistil_mask": pistil_comp_mask,
                "pistil_contour": pistil_cnt,
                "pistil_ellipse": pistil_ellipse,
                "center_pistil": center_pistil,
                "pistil_area": float(cv2.contourArea(pistil_cnt)) if pistil_cnt is not None else 0.0,
                "dir_info": dir_info
            }

            flower_info["pistils"].append(pistil_info)

        all_results.append(flower_info)

    return flower_mask_all, pistil_mask_all, all_results


def draw_all_results(img_bgr, all_results, draw_thickness=4):
    """
    Vẽ tất cả ellipse, tâm và véc tơ.
    """
    out = img_bgr.copy()

    flower_color = (255, 255, 0)   # cyan trong BGR
    pistil_color = (0, 0, 255)     # đỏ
    vector_color = (255, 0, 0)     # xanh dương

    center_radius = draw_thickness + 2

    for flower_info in all_results:
        flower_ellipse = flower_info["flower_ellipse"]
        center_flower = flower_info["center_flower"]

        if flower_ellipse is not None:
            cv2.ellipse(out, flower_ellipse, flower_color, draw_thickness, cv2.LINE_AA)

        if center_flower is not None:
            cv2.circle(out, center_flower, center_radius, flower_color, -1, cv2.LINE_AA)

        for pistil_info in flower_info["pistils"]:
            pistil_ellipse = pistil_info["pistil_ellipse"]
            center_pistil = pistil_info["center_pistil"]
            dir_info = pistil_info["dir_info"]

            if pistil_ellipse is not None:
                cv2.ellipse(out, pistil_ellipse, pistil_color, draw_thickness, cv2.LINE_AA)

            if center_pistil is not None:
                cv2.circle(out, center_pistil, center_radius, pistil_color, -1, cv2.LINE_AA)

            if dir_info is not None and center_flower is not None and center_pistil is not None:
                cv2.arrowedLine(
                    out,
                    center_pistil,
                    center_flower,
                    vector_color,
                    draw_thickness + 1,
                    cv2.LINE_AA,
                    tipLength=0.22
                )

    return out


def read_input_image():
    img_name = "5.png"
    root_path = "tl1Img"
    path = os.path.join(".", root_path, img_name)

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")

    return img, path, img_name


def main():
    img, img_path, img_name = read_input_image()
    OUTPUT_DIR = f"./saveImg/output_flower_detect{img_name}"
    print(f"Da doc anh: {img_path}")
    print(f"FLOWER_SEGMENT_METHOD = {FLOWER_SEGMENT_METHOD}")

    t0 = time.perf_counter()

    flower_mask_all, pistil_mask_all, all_results = analyze_all_flowers_and_pistils(
        img,
        flower_min_component_area=FLOWER_MIN_COMPONENT_AREA,
        pistil_min_component_area=PISTIL_MIN_COMPONENT_AREA,
        flower_segment_method=FLOWER_SEGMENT_METHOD
    )

    result_img = draw_all_results(
        img,
        all_results,
        draw_thickness=DRAW_THICKNESS
    )

    t1 = time.perf_counter()
    print(f"latency: {t1 - t0:.6f} s")

    total_flowers = len(all_results)
    total_pistils = sum(len(f["pistils"]) for f in all_results)

    print(f"tong so flower masks = {total_flowers}")
    print(f"tong so pistil masks = {total_pistils}")

    for flower_info in all_results:
        print("-" * 60)
        print(f"flower #{flower_info['flower_idx']}")
        print(f"  flower_area   = {flower_info['flower_area']:.2f}")
        print(f"  center_flower = {flower_info['center_flower']}")
        print(f"  flower_ellipse = {flower_info['flower_ellipse']}")

        if len(flower_info["pistils"]) == 0:
            print("  Khong tim thay pistil mask nao trong flower nay.")
            continue

        for pistil_info in flower_info["pistils"]:
            print(f"  pistil #{pistil_info['pistil_idx']}")
            print(f"    pistil_area   = {pistil_info['pistil_area']:.2f}")
            print(f"    center_pistil = {pistil_info['center_pistil']}")
            print(f"    pistil_ellipse = {pistil_info['pistil_ellipse']}")

            dir_info = pistil_info["dir_info"]
            if dir_info is not None:
                print(f"    dx = {dir_info['dx']:.3f}, dy = {dir_info['dy']:.3f}")
                print(f"    theta_img_deg  = {dir_info['theta_img_deg']:.3f}")
                print(f"    theta_math_deg = {dir_info['theta_math_deg']:.3f}")
                print(f"    theta_axis_deg = {dir_info['theta_axis_deg']:.3f}")
            else:
                print("    Khong tinh duoc huong.")

    if AUTO_SAVE_IMAGES:
        save_output_images(
            OUTPUT_DIR,
            original_img=img,
            result_img=result_img,
            flower_mask=flower_mask_all,
            pistil_mask=pistil_mask_all
        )
        print(f"Da luu anh vao thu muc: {OUTPUT_DIR}")

        # Ví dụ gọi riêng hàm lưu mask nếu bạn muốn:
        # save_mask_image(flower_mask_all, os.path.join(OUTPUT_DIR, "flower_mask_inv.png"), invert=True)
        # save_mask_image(pistil_mask_all, os.path.join(OUTPUT_DIR, "pistil_mask_big.png"), scale=2.0)
        # save_mask_image(flower_mask_all, os.path.join(OUTPUT_DIR, "flower_mask_bgr.png"), as_bgr=True)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(flower_mask_all, cmap="gray")
    plt.title("All flower masks")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(pistil_mask_all, cmap="gray")
    plt.title("All pistil masks (from INV flower mask)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Result - all ellipses, centers and directions")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()