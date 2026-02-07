# import torch
from PIL import Image, ImageDraw, ImageFont
from functools import cached_property
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import matplotlib
import cv2
from skimage.morphology import medial_axis, skeletonize
from pathlib import Path
import json
import torch
from utils import *


SALAMANDRA_DB_ROOT = '/data/Env/Data/Slmndr/'
LOOKUP_RADIUS_FACTOR = .5
ANGLE_LOOKUP_RANGE = 40

def dbg():
    import debugpy
    from IPython import get_ipython
    ip = get_ipython()
    ip.kernel.post_handler_hook = lambda self: self.shell.register_debugger_sigint() # patch ipython kernel to allow debug

    debug_port = 5678
    found_port = False
    while not found_port:
        try:
            debugpy.listen(debug_port)
        except Exception as ex:
            print(f'Can\'t start the debugger on port {debug_port}. Trying another port ...')
            debug_port += 1
        else:
            found_port = True
            print(f'Waiting for debugger on port {debug_port}')
        debugpy.wait_for_client()


def overlay_masks(
    image,
    masks,
    show_numbers=False,
    start_index=0,
    text_color=(255, 255, 255),
    text_outline=(0, 0, 0),
    font=None,
):
    image = image.convert("RGBA")
    masks = 255 * masks.detach().cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]
    color = colors[0]
    #for mask, color in zip(masks, colors):
    for mask in masks:
        mask = Image.fromarray(mask.squeeze())
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    if show_numbers:
        draw = ImageDraw.Draw(image)
        if font is None:
            font = ImageFont.load_default()
        for idx, mask in enumerate(masks):
            mask_u8 = mask.squeeze()
            center = mask_barycenter(mask_u8)
            if center is None:
                continue
            cx, cy = center
            label = str(idx + start_index)
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
    return image

def crop_mask(image, mask):
    img_np = np.array(image)
    img_np[~mask] = 0
    return img_np

def bbox_buffer(bbox, buffer, w, h):
    x1, y1, x2, y2 = bbox
    xbuffer = (x2 - x1) * buffer
    ybuffer = (y2 - y1) * buffer
    x1 = max(0, x1 - xbuffer)
    y1 = max(0, y1 - xbuffer)
    x2 = min(w, x2 + xbuffer)
    y2 = min(h, y2 + ybuffer)
    return np.array([x1, y1, x2, y2])

def crop_to_bbox(img, bbox):
    h, w = img.shape[:2]

    x1, y1, x2, y2 = bbox  # may be float

    # convert to int (pixel indices)
    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))

    # clip to image bounds
    x1 = np.clip(x1, 0, w)
    x2 = np.clip(x2, 0, w)
    y1 = np.clip(y1, 0, h)
    y2 = np.clip(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox after clipping")

    return img[y1:y2, x1:x2]

def distance_from_curve(b, x, y):
    """Return the minimum Euclidean distance from (x, y) to a curve.

    Supports:
    - polyline: shape (N, 2)
    - cubic Bezier segments: shape (M, 4, 2) or a single segment (4, 2)
    """
    if b is None:
        return float("inf")

    pts = np.asarray(b, dtype=np.float64)
    if pts.size == 0:
        return float("inf")

    px = float(x)
    py = float(y)
    p = np.array([px, py], dtype=np.float64)

    # Polyline input (N, 2)
    if pts.ndim == 2 and pts.shape[1] == 2:
        if pts.shape[0] == 1:
            d = pts[0] - p
            return float(np.sqrt(d[0] * d[0] + d[1] * d[1]))
        a = pts[:-1]
        c = pts[1:]
        ab = c - a
        ap = p - a
        denom = np.sum(ab * ab, axis=1)
        denom = np.where(denom > 0.0, denom, 1.0)
        t = np.sum(ap * ab, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)[:, None]
        closest = a + t * ab
        diff = closest - p
        dist2 = np.sum(diff * diff, axis=1)
        return float(np.sqrt(np.min(dist2)))

    # Single cubic Bezier segment (4, 2)
    if pts.ndim == 2 and pts.shape == (4, 2):
        pts = pts[None, ...]

    # Cubic Bezier segments (M, 4, 2)
    if pts.ndim == 3 and pts.shape[1] == 4 and pts.shape[2] == 2:
        samples = 64
        t = np.linspace(0.0, 1.0, samples)
        mt = 1.0 - t
        min_dist2 = float("inf")
        for seg in pts:
            p0, p1, p2, p3 = seg
            curve = (
                (mt ** 3)[:, None] * p0
                + (3.0 * (mt ** 2) * t)[:, None] * p1
                + (3.0 * mt * (t ** 2))[:, None] * p2
                + (t ** 3)[:, None] * p3
            )
            diff = curve - p
            dist2 = np.sum(diff * diff, axis=1)
            min_dist2 = min(min_dist2, float(np.min(dist2)))
        return float(np.sqrt(min_dist2))

    raise ValueError("Unsupported curve format for distance_from_curve")

def curve_side(b, x, y):
    """Return +1 if point is to the right of the curve direction, -1 if left.

    For Bezier curves, uses the closest sampled point and local tangent.
    For polylines, uses the closest segment and its direction.
    """
    if b is None:
        return 0

    pts = np.asarray(b, dtype=np.float64)
    if pts.size == 0:
        return 0

    p = np.array([float(x), float(y)], dtype=np.float64)

    # Polyline input (N, 2)
    if pts.ndim == 2 and pts.shape[1] == 2:
        if pts.shape[0] < 2:
            return 0
        a = pts[:-1]
        c = pts[1:]
        ab = c - a
        ap = p - a
        denom = np.sum(ab * ab, axis=1)
        denom = np.where(denom > 0.0, denom, 1.0)
        t = np.sum(ap * ab, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)[:, None]
        closest = a + t * ab
        diff = closest - p
        dist2 = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(dist2))
        tangent = ab[idx]
        if np.allclose(tangent, 0.0):
            return 0
        v = p - closest[idx]
        cross = tangent[0] * v[1] - tangent[1] * v[0]
        return 1 if cross < 0.0 else -1

    # Single cubic Bezier segment (4, 2)
    if pts.ndim == 2 and pts.shape == (4, 2):
        pts = pts[None, ...]

    # Cubic Bezier segments (M, 4, 2)
    if pts.ndim == 3 and pts.shape[1] == 4 and pts.shape[2] == 2:
        samples = 64
        t = np.linspace(0.0, 1.0, samples)
        mt = 1.0 - t
        min_dist2 = float("inf")
        best_tangent = None
        best_point = None
        for seg in pts:
            p0, p1, p2, p3 = seg
            curve = (
                (mt ** 3)[:, None] * p0
                + (3.0 * (mt ** 2) * t)[:, None] * p1
                + (3.0 * mt * (t ** 2))[:, None] * p2
                + (t ** 3)[:, None] * p3
            )
            diff = curve - p
            dist2 = np.sum(diff * diff, axis=1)
            idx = int(np.argmin(dist2))
            if float(dist2[idx]) < min_dist2:
                min_dist2 = float(dist2[idx])
                best_point = curve[idx]
                tt = t[idx]
                omt = 1.0 - tt
                # First derivative of cubic Bezier
                best_tangent = (
                    3.0 * (omt ** 2) * (p1 - p0)
                    + 6.0 * omt * tt * (p2 - p1)
                    + 3.0 * (tt ** 2) * (p3 - p2)
                )
        if best_tangent is None or best_point is None or np.allclose(best_tangent, 0.0):
            return 0
        v = p - best_point
        cross = best_tangent[0] * v[1] - best_tangent[1] * v[0]
        return 1 if cross < 0.0 else -1

    raise ValueError("Unsupported curve format for curve_side")

def match_detections(spot_a, spot_b, ratio=0.75, ransac_thresh=3.0):
    if spot_a.descriptor is None or spot_b.descriptor is None:
        return []
    if spot_a.descriptor.descriptors is None or spot_b.descriptor.descriptors is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(
        spot_a.descriptor.descriptors,
        spot_b.descriptor.descriptors,
        k=2,
    )
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 4:
        return good
    src_pts = np.float32(
        [spot_a.descriptor.keypoints[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [spot_b.descriptor.keypoints[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return []
    inliers = [m for m, keep in zip(good, mask.ravel()) if keep]
    return inliers

    
class Detection:
    def __init__(self, bbox, mask, score, image, masked_image, yellow_image):
        self.bbox = bbox
        self.mask = mask
        self.score = score
        self.image = image
        self.masked_image = masked_image
        self.yellow_image = yellow_image

    def store(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "mask.npy", self.mask)
        np.save(path / "bbox.npy", self.bbox)
        (path / "meta.json").write_text(json.dumps({"score": self.score}))
        Image.fromarray(self.image).save(path / "image.png")
        Image.fromarray(self.masked_image).save(path / "masked_image.png")
        np.save(path / "yellow_image.npy", self.yellow_image)

    @staticmethod
    def from_detection_result(original_image, detection_result, idx, bufferRatio = .1):
        width = original_image.width
        height = original_image.height
        mask = detection_result['masks'][idx].detach().cpu().squeeze().numpy()
        bbox = bbox_buffer(detection_result['boxes'][idx].cpu().numpy(), bufferRatio, width, height)
        score = float(detection_result['scores'][0].cpu())
        mask = crop_to_bbox(mask, bbox)
        image = crop_to_bbox(np.array(original_image), bbox)
        masked_image = crop_mask(image, mask)
        yellow_image = normalized_yellow_intensity(image)
        return Detection(bbox, mask, score, image, masked_image, yellow_image)

    @staticmethod
    def from_prompt_results(original_image, detection_result):
        return [Detection.from_detection_result(original_image, detection_result, i) for i in range(detection_result['scores'].shape[0])]

    @staticmethod
    def load(path, load_images=False):
        path = Path(path)
        mask = np.load(path / "mask.npy") if load_images else None
        bbox = np.load(path / "bbox.npy")
        meta = json.loads((path / "meta.json").read_text())
        image = Image.open(path / "image.png") if load_images else None
        masked_image = Image.open(path / "masked_image.png") if load_images else None
        yellow_path = path / "yellow_image.npy"
        yellow_image = np.load(yellow_path) if load_images and yellow_path.exists() else None
        return Detection(bbox, mask, meta.get("score"), image, masked_image, yellow_image)

class Descriptor:
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

    def store(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        keypoints = self.keypoints or []
        kp_data = np.array(
            [
                (
                    kp.pt[0],
                    kp.pt[1],
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id,
                )
                for kp in keypoints
            ],
            dtype=np.float32,
        )
        np.save(path / "keypoints.npy", kp_data)
        if self.descriptors is not None:
            np.save(path / "descriptors.npy", self.descriptors)

    @staticmethod
    def load(path, load_images=False):
        path = Path(path)
        kp_data = np.load(path / "keypoints.npy")
        keypoints = [
            cv2.KeyPoint(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                int(row[5]),
                int(row[6]),
            )
            for row in kp_data
        ]
        descriptors_path = path / "descriptors.npy"
        descriptors = np.load(descriptors_path) if descriptors_path.exists() else None
        return Descriptor(keypoints, descriptors)

    @staticmethod
    def from_detection(detection):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute((255 * detection.yellow_image).astype('uint8'), None)
        return Descriptor(keypoints, descriptors)


class Spot:
    def __init__(self, detection, descriptor):
        self.detection = detection
        self.descriptor = descriptor

    def store(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.detection.store(path / "detection")
        if self.descriptor is not None:
            self.descriptor.store(path / "descriptor")

    @staticmethod
    def load(path, load_images=False):
        path = Path(path)
        detection = Detection.load(path / "detection", load_images=load_images)
        descriptor_path = path / "descriptor"
        if descriptor_path.exists():
            descriptor = Descriptor.load(descriptor_path, load_images=load_images)
        else:
            # Backwards compatibility with older spot storage layout.
            descriptor = Descriptor.load(path, load_images=load_images)
        return Spot(detection, descriptor)

    @staticmethod
    def from_detection(detection):
        descriptor = Descriptor.from_detection(detection)
        return Spot(detection, descriptor)

class Tail:
    def __init__(self, detection, descriptor):
        self.detection = detection
        self.descriptor = descriptor

    def store(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.detection.store(path / "detection")
        if self.descriptor is not None:
            self.descriptor.store(path / "descriptor")

    @staticmethod
    def load(path, load_images=False):
        path = Path(path)
        detection = Detection.load(path / "detection", load_images=load_images)
        descriptor_path = path / "descriptor"
        if descriptor_path.exists():
            descriptor = Descriptor.load(descriptor_path, load_images=load_images)
        else:
            descriptor = Descriptor.load(path, load_images=load_images)
        return Tail(detection, descriptor)

    @staticmethod
    def from_detection(detection):
        descriptor = Descriptor.from_detection(detection)
        return Tail(detection, descriptor)


class Body:
    def __init__(self, detection, descriptor, scale=None, dist=None, skel_thin=None, bezier_curve=None):
        self.detection = detection
        self.descriptor = descriptor
        self.scale = scale
        self.dist = dist
        self.skel_thin = skel_thin
        self.bezier_curve = bezier_curve

    def store(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.detection.store(path / "detection")
        if self.descriptor is not None:
            self.descriptor.store(path / "descriptor")
        if self.scale is not None:
            np.save(path / "scale.npy", np.asarray(self.scale, dtype=np.float64))
        if self.dist is not None:
            np.save(path / "dist.npy", self.dist)
        if self.skel_thin is not None:
            np.save(path / "skel_thin.npy", self.skel_thin)
        if self.bezier_curve is not None:
            np.save(path / "bezier_curve.npy", np.asarray(self.bezier_curve, dtype=np.float64))

    @staticmethod
    def load(path, load_images=False):
        path = Path(path)
        detection_path = path / "detection"
        if detection_path.exists():
            detection = Detection.load(detection_path, load_images=load_images)
        else:
            # Backwards compatibility with older body storage layout.
            detection = Detection.load(path, load_images=load_images)
        descriptor_path = path / "descriptor"
        if descriptor_path.exists():
            descriptor = Descriptor.load(descriptor_path, load_images=load_images)
        else:
            legacy_keypoints = path / "keypoints.npy"
            descriptor = (
                Descriptor.load(path, load_images=load_images)
                if legacy_keypoints.exists()
                else None
            )
        scale_path = path / "scale.npy"
        if scale_path.exists():
            scale = np.load(scale_path)
            if np.ndim(scale) == 0:
                scale = float(scale)
        else:
            scale = None
        dist_path = path / "dist.npy"
        dist = np.load(dist_path) if dist_path.exists() else None
        skel_path = path / "skel_thin.npy"
        skel_thin = np.load(skel_path) if skel_path.exists() else None
        bezier_path = path / "bezier_curve.npy"
        bezier_curve = np.load(bezier_path) if bezier_path.exists() else None
        return Body(detection, descriptor, scale, dist, skel_thin, bezier_curve)

    @staticmethod
    def from_detection(detection):
        descriptor = Descriptor.from_detection(detection)
        dist = mask_distance_transform(detection.mask)
        scale = np.quantile(dist[dist > 0], .9)
        mask_deep = dist * (dist > scale)
        skel_thin = skeletonize(mask_deep)
        pp = trace_ridges(skel_thin, LOOKUP_RADIUS_FACTOR * scale, ANGLE_LOOKUP_RANGE)
        b = bezier_from_polyline(pp)
        return Body(detection, descriptor, scale, dist, skel_thin, b)


class Salamandra:
    def __init__(self, path, original_image, body, spots, tail=None):
        self.path = path
        self.original_image = original_image
        self.body = body
        self.spots = spots
        self.tail = tail
    
    def store(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(json.dumps({"path": str(self.path) if self.path is not None else None}))
        self.original_image.save(path / "original.png")
        self.body.store(path / "body")
        if self.tail is not None:
            self.tail.store(path / "tail")
        for idx, spot in enumerate(self.spots):
            spot.store(path / f"spot_{idx}")

    @staticmethod
    def load(path, load_images=False):
        path = Path(path)
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            stored_path = meta.get("path")
        else:
            stored_path = None
        original_image = Image.open(path / "original.png") if load_images else None
        body = Body.load(path / "body", load_images=load_images)
        tail = None
        tail_path = path / "tail"
        if tail_path.exists():
            tail = Tail.load(tail_path, load_images=load_images)
        spot_dirs = list(sorted(p for p in path.glob("spot_*") if p.is_dir()))
        spots = [None for _ in enumerate(spot_dirs)]
        for spot_dir in sorted(p for p in path.glob("spot_*") if p.is_dir()):
            spot_id = int(spot_dir.name[5:])
            spots[spot_id] = Spot.load(spot_dir, load_images=load_images)
        return Salamandra(stored_path, original_image, body, spots, tail=tail)

    @staticmethod
    def from_image(detector, image_path):
        image = Image.open(image_path)
        salamandras = detector.detect(image, 'salamandra')
        results = []
        for salamandra in salamandras:
            spot_detections = detector.detect(Image.fromarray(salamandra.image), 'spot')
            tail_detections = detector.detect(Image.fromarray(salamandra.masked_image), 'tail')
            tail = (
                Tail.from_detection(max(tail_detections, key=lambda d: d.score))
                if tail_detections
                else None
            )
            results.append(
                Salamandra(
                    image_path,
                    image,
                    Body.from_detection(salamandra),
                    [Spot.from_detection(spot) for spot in spot_detections],
                    tail=tail,
                )
            )
        return results



# def detections_from_prompt_result(original_image, detection_result, euqalizeHist = False):
#     return [Detection(original_image, detection_result, i) for i in range(detection_result['scores'].shape[0])]

class Detector:
    def __init__(self):
        self.processor = None

    def load(self):
        model = build_sam3_image_model() # checkpoint_path='/data/Env/Dev/T/sam3.pt')
        self.processor = Sam3Processor(model)

    def unload(self):
        self.processor = None
        self.model = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def detect(self, image, prompt):
        if self.processor is None:
            self.load()
        inference_state = self.processor.set_image(image.convert("RGB"))
        detection_result = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        return Detection.from_prompt_results(image, detection_result)

class SalamandraDB:
    def __init__(self, root_path=SALAMANDRA_DB_ROOT, load_images=False):
        self.root_path = Path(root_path)
        self.content = {p.name: Salamandra.load(p, load_images=load_images) for p in self.root_path.glob("*") if p.is_dir()}
        self.counter = max((int(k) for k in self.content if k.isnumeric()), default=-1) + 1
 
    @cached_property
    def detector(self):
        return Detector()
        
    def add(self, image_path):
        salamandras = Salamandra.from_image(self.detector, image_path)
        for salamandra in salamandras:
            salamandra.store(self.root_path / str(self.counter))
            self.content[str(self.counter)] = salamandra
            self.counter += 1
        self.detector.unload()
        return salamandras
