import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


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


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.where(labels == largest_id, 255, 0).astype(np.uint8)
    return out


def flower_mask_hsv(img_bgr, keep_largest=True):
    """
    Tách bông hoa vàng bằng HSV.
    Lưu ý: hậu xử lý giữ nhẹ để không làm mất 'lỗ' bên trong mask,
    vì lỗ đó sẽ được xem là contour trong của nhụy.
    """
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Có thể chỉnh lại nếu ảnh khác nhiều
    lower = np.array([15, 60, 30], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    if keep_largest:
        mask = keep_largest_component(mask)

    return mask


def contour_centroid_from_contour(cnt):
    if cnt is None:
        return None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-9:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def find_outer_and_inner_contour_from_flower_mask(flower_mask):
    """
    Tìm contour ngoài và contour trong trực tiếp từ flower_mask.
    - contour ngoài: contour lớn nhất có parent = -1
    - contour trong: contour con lớn nhất của contour ngoài
    """
    work = flower_mask.copy()

    contours, hierarchy = cv2.findContours(
        work,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0 or hierarchy is None:
        return None, None, None

    hierarchy = hierarchy[0]

    # ===== tìm contour ngoài lớn nhất =====
    outer_ids = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    if len(outer_ids) == 0:
        return None, None, None

    outer_id = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))
    outer_cnt = contours[outer_id]

    # ===== tìm các contour con trực tiếp của contour ngoài =====
    child_ids = []
    child = hierarchy[outer_id][2]  # first child
    while child != -1:
        child_ids.append(child)
        child = hierarchy[child][0]  # next sibling

    inner_cnt = None
    if len(child_ids) > 0:
        inner_id = max(child_ids, key=lambda i: cv2.contourArea(contours[i]))
        if cv2.contourArea(contours[inner_id]) > 5:
            inner_cnt = contours[inner_id]

    # ===== fallback nếu hierarchy không ra contour trong như mong muốn =====
    hole_mask = np.zeros_like(flower_mask)
    if inner_cnt is None:
        filled_outer = np.zeros_like(flower_mask)
        cv2.drawContours(filled_outer, [outer_cnt], -1, 255, thickness=-1)

        # phần nằm trong contour ngoài nhưng không nằm trong mask gốc => lỗ bên trong
        hole_mask = cv2.subtract(filled_outer, flower_mask)

        hole_contours, _ = cv2.findContours(
            hole_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(hole_contours) > 0:
            inner_cnt = max(hole_contours, key=cv2.contourArea)
            if cv2.contourArea(inner_cnt) <= 5:
                inner_cnt = None
    else:
        cv2.drawContours(hole_mask, [inner_cnt], -1, 255, thickness=-1)

    return outer_cnt, inner_cnt, hole_mask


def draw_boxes_and_centers_on_original(img_bgr, outer_cnt, inner_cnt):
    """
    Ảnh hiển thị 1:
    - vẽ contour ngoài, contour trong
    - vẽ bounding box
    - vẽ tâm flower, tâm nhụy
    """
    out = img_bgr.copy()

    center_flower = None
    center_pistil = None

    if outer_cnt is not None:
        cv2.drawContours(out, [outer_cnt], -1, (0, 255, 0), 2)

        x, y, w, h = cv2.boundingRect(outer_cnt)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center_flower = contour_centroid_from_contour(outer_cnt)
        if center_flower is not None:
            cx, cy = center_flower
            cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(
                out, "Flower C", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
            )

    if inner_cnt is not None:
        cv2.drawContours(out, [inner_cnt], -1, (255, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(inner_cnt)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)

        center_pistil = contour_centroid_from_contour(inner_cnt)
        if center_pistil is not None:
            cx, cy = center_pistil
            cv2.circle(out, (cx, cy), 5, (255, 0, 255), -1)
            cv2.putText(
                out, "Pistil C", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA
            )

    return out, center_flower, center_pistil


def compute_direction_info(center_flower, center_pistil):
    if center_flower is None or center_pistil is None:
        return None

    cx_f, cy_f = center_flower
    cx_p, cy_p = center_pistil

    dx = float(cx_p - cx_f)
    dy = float(cy_p - cy_f)

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


def draw_vector_on_original(img_bgr, outer_cnt, inner_cnt, center_flower, center_pistil):
    """
    Ảnh hiển thị 2:
    - vẽ lại contour ngoài, contour trong
    - vẽ véc tơ hướng từ tâm flower -> tâm nhụy
    """
    out = img_bgr.copy()

    if outer_cnt is not None:
        cv2.drawContours(out, [outer_cnt], -1, (0, 255, 0), 2)

    if inner_cnt is not None:
        cv2.drawContours(out, [inner_cnt], -1, (255, 0, 255), 2)

    if center_flower is not None:
        cv2.circle(out, center_flower, 5, (0, 255, 0), -1)

    if center_pistil is not None:
        cv2.circle(out, center_pistil, 5, (255, 0, 255), -1)

    dir_info = compute_direction_info(center_flower, center_pistil)
    if dir_info is None:
        return out, None

    cx_f, cy_f = center_flower
    cx_p, cy_p = center_pistil

    # véc tơ hướng: từ tâm flower -> tâm nhụy
    cv2.arrowedLine(
        out,
        (cx_f, cy_f),
        (cx_p, cy_p),
        (0, 0, 255),
        3,
        tipLength=0.25
    )

    txt1 = f"theta_img  = {dir_info['theta_img_deg']:.2f} deg"
    txt2 = f"theta_math = {dir_info['theta_math_deg']:.2f} deg"
    txt3 = f"theta_axis = {dir_info['theta_axis_deg']:.2f} deg"

    cv2.putText(out, txt1, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, txt2, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, txt3, (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return out, dir_info


def read_input_image():
    img_name = "4.png"
    root_path = "tl1Img"
    path = os.path.join(".", root_path, img_name)

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")

    return img, path


def main():
    img, img_path = read_input_image()
    print(f"Da doc anh: {img_path}")

    gray = to_gray(img)

    t0 = time.perf_counter()

    # ====== tách mask bông hoa ======
    flower_mask = flower_mask_hsv(img, keep_largest=True)
    flower_only = apply_mask_to_bgr(img, flower_mask)

    # ====== tìm contour ngoài và contour trong từ chính flower_mask ======
    outer_cnt, inner_cnt, hole_mask = find_outer_and_inner_contour_from_flower_mask(flower_mask)

    t1 = time.perf_counter()
    print(f"latency: {t1 - t0:.6f} s")

    if outer_cnt is not None:
        print(f"outer contour area  = {cv2.contourArea(outer_cnt):.2f}")
    else:
        print("Khong tim thay contour ngoai.")

    if inner_cnt is not None:
        print(f"inner contour area  = {cv2.contourArea(inner_cnt):.2f}")
    else:
        print("Khong tim thay contour trong (nhuy).")

    # ====== ảnh hiển thị 1: box + tâm ======
    img_boxes, center_flower, center_pistil = draw_boxes_and_centers_on_original(
        img, outer_cnt, inner_cnt
    )

    print("center_flower =", center_flower)
    print("center_pistil =", center_pistil)

    # ====== ảnh hiển thị 2: véc tơ hướng ======
    img_vector, dir_info = draw_vector_on_original(
        img, outer_cnt, inner_cnt, center_flower, center_pistil
    )

    if dir_info is not None:
        print(f"dx = {dir_info['dx']:.3f}, dy = {dir_info['dy']:.3f}")
        print(f"theta_img_deg  = {dir_info['theta_img_deg']:.3f}")
        print(f"theta_math_deg = {dir_info['theta_math_deg']:.3f}")
        print(f"theta_axis_deg = {dir_info['theta_axis_deg']:.3f}")

    # ====== hiển thị ======
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(flower_mask, cmap="gray")
    plt.title("Flower mask")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Image 1 - Bounding boxes and centers")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(img_vector, cv2.COLOR_BGR2RGB))
    plt.title("Image 2 - Direction vector on original")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ====== hiển thị thêm hole mask nếu muốn debug ======
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(flower_only, cv2.COLOR_BGR2RGB))
    plt.title("Flower only")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hole_mask, cmap="gray")
    plt.title("Inner hole mask (pistil region)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()