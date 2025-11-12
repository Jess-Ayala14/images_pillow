import cv2
import numpy as np

def enhance_image(input_path, output_path,
                  brightness=0, contrast=1.0, sharpness=1.5,
                  saturation=1.0, gamma=1.0, denoise=0,
                  color_temp=0, edge_mark=0):
    img = cv2.imread(input_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # --- Gamma correction ---
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)

    # --- Brightness & Contrast ---
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # --- CLAHE (mejor contraste local) ---
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # --- Sharpness (Unsharp Mask) ---
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, sharpness, blurred, -(sharpness - 1), 0)

    # --- Saturation ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * saturation, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Temperatura de color ---
    if color_temp != 0:
        if color_temp > 0:
            img[:, :, 2] = np.clip(img[:, :, 2] + color_temp, 0, 255)  # cálido
        else:
            img[:, :, 0] = np.clip(img[:, :, 0] + abs(color_temp), 0, 255)  # frío

    # --- Marcado de contornos ---
    if edge_mark > 0:
        edges = cv2.Canny(img, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 1, edges_colored, edge_mark * 0.5, 0)

    # --- Denoise automático (0-4) ---
    if denoise > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, h=denoise*5, hColor=denoise*5)

    cv2.imwrite(output_path, img)
