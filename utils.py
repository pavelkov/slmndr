import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def mask_barycenter(mask):
    """Return (x, y) barycenter of non-zero mask pixels as floats."""
    mask_bool = np.asarray(mask) > 0
    coords = np.argwhere(mask_bool)
    if coords.size == 0:
        return None
    y_mean = float(coords[:, 0].mean())
    x_mean = float(coords[:, 1].mean())
    return (x_mean, y_mean)


def overlay_points(image, points, color=(255, 0, 0), radius=3, alpha=0.8):
    """Return image with points drawn as filled circles."""
    if points is None:
        return image
    pts = np.asarray(points)
    if pts.size == 0:
        return image
    pts = np.reshape(pts, (-1, 2))

    if isinstance(image, Image.Image):
        base = image.copy()
    else:
        base = Image.fromarray(np.asarray(image))

    if base.mode != "RGBA":
        base = base.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (int(color[0]), int(color[1]), int(color[2]), int(255 * alpha))
    for x, y in pts:
        x = float(x)
        y = float(y)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    out = Image.alpha_composite(base, overlay)

    if isinstance(image, Image.Image):
        return out
    return np.asarray(out)


def overlay_spot_masks_with_numbers(
    image,
    spots,
    color=(255, 0, 0),
    alpha=0.35,
    show_numbers=True,
    text_color=(255, 255, 255),
    text_outline=(0, 0, 0),
    font=None,
    start_index=0,
):
    """Overlay spot masks and index labels on the full image."""
    if spots is None:
        return image

    if image is None and hasattr(spots, "spots"):
        salamandra = spots
        spots = salamandra.spots
        if getattr(salamandra, "body", None) is not None and getattr(salamandra.body, "image", None) is not None:
            image = salamandra.body.image
        elif getattr(salamandra, "original_image", None) is not None:
            image = salamandra.original_image

    if image is None:
        raise ValueError("image must be provided when spots is a list")

    if isinstance(image, Image.Image):
        base = image.copy()
    else:
        base = Image.fromarray(np.asarray(image))

    if base.mode != "RGBA":
        base = base.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    labels = []
    if show_numbers and font is None:
        font = ImageFont.load_default()

    for idx, spot in enumerate(spots):
        if spot is None:
            continue
        detection = spot.detection if hasattr(spot, "detection") else spot
        mask = getattr(detection, "mask", None)
        bbox = getattr(detection, "bbox", None)
        if mask is None or bbox is None:
            continue

        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1i = int(np.floor(x1))
        y1i = int(np.floor(y1))
        x2i = int(np.ceil(x2))
        y2i = int(np.ceil(y2))

        mask_img = Image.fromarray(mask_u8, mode="L")
        alpha_img = Image.new("L", base.size, 0)
        alpha_img.paste(mask_img, (x1i, y1i))
        alpha_scaled = alpha_img.point(lambda v: int(v * alpha))
        spot_overlay = Image.new("RGBA", base.size, color + (0,))
        spot_overlay.putalpha(alpha_scaled)
        overlay = Image.alpha_composite(overlay, spot_overlay)

        label_pos = None
        if np.any(mask_u8 > 0):
            dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
            if dist.size > 0 and np.isfinite(dist).any():
                y_peak, x_peak = np.unravel_index(int(np.argmax(dist)), dist.shape)
                if dist[y_peak, x_peak] > 0:
                    label_pos = (x1i + float(x_peak), y1i + float(y_peak))

        if label_pos is None:
            center = mask_barycenter(mask_u8)
            if center is None:
                cx = (x1i + x2i) / 2.0
                cy = (y1i + y2i) / 2.0
            else:
                cx = x1i + float(center[0])
                cy = y1i + float(center[1])
        else:
            cx, cy = label_pos

        if show_numbers:
            labels.append((idx + start_index, cx, cy))

    if show_numbers:
        draw = ImageDraw.Draw(overlay)
        for label_num, cx, cy in labels:
            label = str(label_num)
            if hasattr(draw, "textbbox"):
                text_box = draw.textbbox((0, 0), label, font=font)
                text_w = text_box[2] - text_box[0]
                text_h = text_box[3] - text_box[1]
            else:
                text_w, text_h = draw.textsize(label, font=font)
            text_x = cx - text_w / 2.0
            text_y = cy - text_h / 2.0

            if text_outline is not None:
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        if ox == 0 and oy == 0:
                            continue
                        draw.text(
                            (text_x + ox, text_y + oy),
                            label,
                            fill=text_outline,
                            font=font,
                        )
            draw.text((text_x, text_y), label, fill=text_color, font=font)

    out = Image.alpha_composite(base, overlay)
    if isinstance(image, Image.Image):
        return out
    return np.asarray(out)

def overlay_spot_masks_with_distance_red(
    image,
    spots,
    alpha=0.8,
    show_numbers=True,
    text_color=(255, 255, 255),
    text_outline=(0, 0, 0),
    font=None,
    start_index=0,
):
    """Overlay distance transform in red with alpha proportional to distance."""
    if spots is None:
        return image

    if image is None and hasattr(spots, "spots"):
        salamandra = spots
        spots = salamandra.spots
        if getattr(salamandra, "body", None) is not None and getattr(salamandra.body, "image", None) is not None:
            image = salamandra.body.image
        elif getattr(salamandra, "original_image", None) is not None:
            image = salamandra.original_image

    if image is None:
        raise ValueError("image must be provided when spots is a list")

    if isinstance(image, Image.Image):
        base = image.copy()
    else:
        base = Image.fromarray(np.asarray(image))

    if base.mode != "RGBA":
        base = base.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    labels = []
    if show_numbers and font is None:
        font = ImageFont.load_default()

    for idx, spot in enumerate(spots):
        if spot is None:
            continue
        detection = spot.detection if hasattr(spot, "detection") else spot
        mask = getattr(detection, "mask", None)
        bbox = getattr(detection, "bbox", None)
        if mask is None or bbox is None:
            continue

        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1i = int(np.floor(x1))
        y1i = int(np.floor(y1))
        x2i = int(np.ceil(x2))
        y2i = int(np.ceil(y2))

        if np.any(mask_u8 > 0):
            dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
            dist_max = float(dist.max())
            if dist_max > 0:
                dist_norm = np.clip(dist / dist_max * 255.0, 0, 255).astype(np.uint8)
            else:
                dist_norm = np.zeros_like(mask_u8, dtype=np.uint8)
        else:
            dist_norm = np.zeros_like(mask_u8, dtype=np.uint8)

        dist_img = Image.fromarray(dist_norm, mode="L")
        alpha_img = Image.new("L", base.size, 0)
        alpha_img.paste(dist_img, (x1i, y1i))
        if alpha != 1.0:
            alpha_img = alpha_img.point(lambda v: int(v * alpha))
        spot_overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
        spot_overlay.putalpha(alpha_img)
        overlay = Image.alpha_composite(overlay, spot_overlay)

        if show_numbers:
            label_pos = None
            if np.any(mask_u8 > 0):
                dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
                if dist.size > 0 and np.isfinite(dist).any():
                    y_peak, x_peak = np.unravel_index(int(np.argmax(dist)), dist.shape)
                    if dist[y_peak, x_peak] > 0:
                        label_pos = (x1i + float(x_peak), y1i + float(y_peak))
            if label_pos is None:
                center = mask_barycenter(mask_u8)
                if center is None:
                    cx = (x1i + x2i) / 2.0
                    cy = (y1i + y2i) / 2.0
                else:
                    cx = x1i + float(center[0])
                    cy = y1i + float(center[1])
            else:
                cx, cy = label_pos
            labels.append((idx + start_index, cx, cy))

    if show_numbers:
        draw = ImageDraw.Draw(overlay)
        for label_num, cx, cy in labels:
            label = str(label_num)
            if hasattr(draw, "textbbox"):
                text_box = draw.textbbox((0, 0), label, font=font)
                text_w = text_box[2] - text_box[0]
                text_h = text_box[3] - text_box[1]
            else:
                text_w, text_h = draw.textsize(label, font=font)
            text_x = cx - text_w / 2.0
            text_y = cy - text_h / 2.0

            if text_outline is not None:
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        if ox == 0 and oy == 0:
                            continue
                        draw.text(
                            (text_x + ox, text_y + oy),
                            label,
                            fill=text_outline,
                            font=font,
                        )
            draw.text((text_x, text_y), label, fill=text_color, font=font)

    out = Image.alpha_composite(base, overlay)
    if isinstance(image, Image.Image):
        return out
    return np.asarray(out)

def concat_images_side_by_side(left, right, bg_color=(0, 0, 0)):
    """Return a new image with two images placed side by side."""
    if isinstance(left, Image.Image):
        left_img = left
    else:
        left_img = Image.fromarray(np.asarray(left))

    if isinstance(right, Image.Image):
        right_img = right
    else:
        right_img = Image.fromarray(np.asarray(right))

    if left_img.mode != "RGBA":
        left_img = left_img.convert("RGBA")
    if right_img.mode != "RGBA":
        right_img = right_img.convert("RGBA")

    width = left_img.width + right_img.width
    height = max(left_img.height, right_img.height)
    canvas = Image.new("RGBA", (width, height), bg_color + (255,))
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width, 0))
    return canvas


def mask_ray_counts(mask, point, bins):
    """Count true mask pixels along rays from point, one count per direction."""
    if bins <= 0:
        raise ValueError("bins must be a positive integer")

    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim > 2:
        mask_bool = mask_bool[..., 0]
    h, w = mask_bool.shape[:2]

    x0, y0 = float(point[0]), float(point[1])
    angles = np.linspace(0.0, 2.0 * np.pi, bins, endpoint=False)
    counts = np.zeros(bins, dtype=int)

    for i, ang in enumerate(angles):
        dx = np.cos(ang)
        dy = np.sin(ang)
        x = x0
        y = y0
        last = None
        while True:
            ix = int(round(x))
            iy = int(round(y))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                break
            if last != (ix, iy):
                if mask_bool[iy, ix]:
                    counts[i] += 1
                last = (ix, iy)
            x += dx
            y += dy

    return counts


def visualize_bin_ray(image, point, bins, bin_index, length=None, color=(0, 255, 0), alpha=0.7, width=2):
    """Overlay a single bin ray on an image for visualization."""
    if bins <= 0:
        raise ValueError("bins must be a positive integer")

    if isinstance(bin_index, (list, tuple, np.ndarray)):
        raise ValueError("bin_index must be a scalar")

    if isinstance(image, Image.Image):
        base = image.copy()
    else:
        base = Image.fromarray(np.asarray(image))

    if base.mode != "RGBA":
        base = base.convert("RGBA")

    w, h = base.size
    x0, y0 = float(point[0]), float(point[1])

    if length is None:
        max_x = max(x0, w - 1 - x0)
        max_y = max(y0, h - 1 - y0)
        length = float(np.hypot(max_x, max_y))
    else:
        length = float(length)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke = (int(color[0]), int(color[1]), int(color[2]), int(255 * alpha))

    idx = int(bin_index)
    if idx < 0 or idx >= bins:
        raise ValueError("bin_index out of range")
    ang = (2.0 * np.pi * idx) / float(bins)
    dx = np.cos(ang)
    dy = np.sin(ang)
    x1 = x0 + dx * length
    y1 = y0 + dy * length
    draw.line((x0, y0, x1, y1), fill=stroke, width=int(width))

    out = Image.alpha_composite(base, overlay)
    if isinstance(image, Image.Image):
        return out
    return np.asarray(out)


def visualize_bin_rays(image, point, bins, bin_index, length=None, color=(0, 255, 0), alpha=0.7, width=2):
    """Backward-compatible wrapper; use visualize_bin_ray."""
    return visualize_bin_ray(
        image,
        point,
        bins,
        bin_index,
        length=length,
        color=color,
        alpha=alpha,
        width=width,
    )


def ridge_from_distance(mask_u8, min_radius=1.0, do_thin=True):
    m = (mask_u8 > 0).astype(np.uint8)

    # distance to background
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)

    # ridge = local maxima of dist (discrete)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dist_dil = cv2.dilate(dist, k)
    ridge = ((dist >= dist_dil - 1e-6) & (dist >= float(min_radius))).astype(np.uint8) * 255

    if do_thin:
        try:
            ridge = cv2.ximgproc.thinning(ridge, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except Exception:
            pass

    return ridge, dist

def viz(img):
    return PIL.Image.fromarray((255 * dist / dist.max()).astype('uint8'))


def yellow_intensity(image, hue_center=30.0, hue_width=20.0, min_sat=0.2, min_val=0.2):
    """
    Compute per-pixel "yellowness" from an RGB image, returning values in [0, 1].

    Args:
        image: RGB numpy array or PIL image.
        hue_center: HSV hue center for yellow in OpenCV units (0-179).
        hue_width: Half-width in hue units for falloff to zero.
        min_sat/min_val: Minimum S/V for non-zero response (soft threshold).
    """
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        return np.zeros_like(rgb, dtype=np.float32)
    if rgb.shape[-1] >= 3:
        rgb = rgb[..., :3]
    else:
        return np.zeros(rgb.shape[:2], dtype=np.float32)

    if np.issubdtype(rgb.dtype, np.floating):
        max_val = float(np.nanmax(rgb)) if rgb.size else 1.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        rgb_u8 = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    dh = np.abs(h - float(hue_center))
    dh = np.minimum(dh, 180.0 - dh)
    hue_score = 1.0 - np.clip(dh / float(hue_width), 0.0, 1.0)

    s_score = np.clip((s - float(min_sat)) / max(1e-6, 1.0 - float(min_sat)), 0.0, 1.0)
    v_score = np.clip((v - float(min_val)) / max(1e-6, 1.0 - float(min_val)), 0.0, 1.0)

    return (hue_score * s_score * v_score).astype(np.float32)


def normalized_yellow_intensity(
    image,
    threshold=0.1,
    q_low=2.0,
    q_high=98.0,
    target_low=0.2,
    target_high=0.98,
):
    """
    Compute yellow_intensity and normalize via linear mapping + gamma correction.

    Quantile mapping: q_low -> target_low, q_high -> target_high.
    Gamma correction: median -> 0.5.
    Values below threshold are set to 0.
    """
    y = yellow_intensity(image).astype(np.float32)
    if y.size == 0:
        return y

    q2 = float(np.nanpercentile(y, q_low))
    q98 = float(np.nanpercentile(y, q_high))
    if not np.isfinite(q2) or not np.isfinite(q98) or q98 <= q2:
        y_lin = np.clip(y, 0.0, 1.0)
    else:
        a = (target_high - target_low) / (q98 - q2)
        b = target_low - a * q2
        y_lin = np.clip(a * y + b, 0.0, 1.0)

    med = float(np.nanmedian(y_lin))
    if med > 0.0 and med < 1.0:
        gamma = np.log(0.5) / np.log(med)
        y_gamma = np.clip(y_lin, 0.0, 1.0) ** gamma
    else:
        y_gamma = y_lin

    if threshold is not None:
        y_gamma = np.where(y_gamma >= float(threshold), y_gamma, 0.0)

    return y_gamma.astype(np.float32)


def show_monochrome(image, cmap="gray", vmin=None, vmax=None, figsize=None, title=None):
    """
    Display a monochrome numpy image in Jupyter (matplotlib if available).

    Args:
        image: 2D numpy array (or array-like).
        cmap: Matplotlib colormap name.
        vmin/vmax: Optional display range.
        figsize: Optional (w, h) for matplotlib.
        title: Optional plot title.
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    try:
        from IPython.display import display

        if np.issubdtype(arr.dtype, np.floating):
            a = arr.astype(np.float32)
            if vmin is None:
                vmin = float(np.nanmin(a)) if a.size else 0.0
            if vmax is None:
                vmax = float(np.nanmax(a)) if a.size else 1.0
            denom = max(1e-6, float(vmax) - float(vmin))
            a = np.clip((a - float(vmin)) / denom, 0.0, 1.0)
            a = (a * 255.0).astype(np.uint8)
        else:
            a = np.clip(arr, 0, 255).astype(np.uint8)

        if cmap is None or str(cmap).lower() in ("gray", "grey", "grays", "greys"):
            img = Image.fromarray(a, mode="L")
        else:
            try:
                from matplotlib import cm

                cmap_obj = cm.get_cmap(cmap)
                rgba = (cmap_obj(a.astype(np.float32) / 255.0) * 255.0).astype(np.uint8)
                img = Image.fromarray(rgba, mode="RGBA")
            except Exception:
                img = Image.fromarray(a, mode="L")

        display(img)
        return
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()
    except Exception:
        raise RuntimeError("No display backend available; install matplotlib or run in Jupyter.")
