import os
import cv2
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
IMG_NAME = "1.png"
ROOT_PATH = "tl1Img"


def main():
    img_path = os.path.join(".", ROOT_PATH, IMG_NAME)
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

    # =========================
    # CHUYỂN ĐỔI KHÔNG GIAN MÀU
    # =========================
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_hls   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    img_luv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LUV)

    # =========================
    # HIỂN THỊ
    # =========================
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Ảnh gốc (RGB hiển thị)")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(img_hsv)
    plt.title("Không gian HSV")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Không gian Grayscale")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(img_lab)
    plt.title("Không gian LAB")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(img_hls)
    plt.title("Không gian HLS")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(img_ycrcb)
    plt.title("Không gian YCrCb")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(img_luv)
    plt.title("Không gian Luv")
    plt.axis("off")

    # ô cuối để trống hoặc có thể ghi chú
    plt.subplot(2, 4, 8)
    plt.text(
        0.5, 0.5,
        "Các ảnh 3 kênh như HSV/LAB/HLS/YCrCb/Luv\n"
        "được hiển thị trực tiếp để quan sát\n"
        "sau khi chuyển đổi không gian màu.",
        ha="center", va="center", fontsize=11
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Tách 3 kênh
    L, A, B = cv2.split(img_lab)
    r,g,b = cv2.split(img_rgb)
    h,s,v = cv2.split(img_hsv)
    # h,l,s = cv2.split(img_hls)
    y,cr,cb = cv2.split(img_ycrcb)
    # =========================
    # HIỂN THỊ
    # =========================
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Ảnh gốc")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(img_lab)
    plt.title("Ảnh trong không gian LAB")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(h, cmap="gray")
    plt.title("Kênh L - Lightness")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(s, cmap="gray")
    plt.title("Kênh A - Green <-> Red")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(v, cmap="gray")
    plt.title("Kênh B - Blue <-> Yellow")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.text(
        0.5, 0.5,
        "LAB:\nL = độ sáng\nA = xanh lá ↔ đỏ\nB = xanh dương ↔ vàng",
        ha="center", va="center", fontsize=13
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()