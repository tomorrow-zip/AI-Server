from mmdet.core import INSTANCE_OFFSET

from collections import defaultdict

import numpy as np
import cv2
import uuid


def detect_process(result, classes):
    pan_results = result["pan_results"]

    ids = np.unique(pan_results)[::-1]
    legal_indices = ids != len(classes)
    ids = ids[legal_indices]
    labels = np.array([id_ % INSTANCE_OFFSET for id_ in ids], dtype=np.int64)
    segms = pan_results[None] == ids[:, None, None]

    detected_labels = {"background": defaultdict(int), "objects": defaultdict(int)}
    detected_objects = {
        "background": defaultdict(defaultdict),
        "objects": defaultdict(defaultdict),
    }

    for segm, label in zip(segms, labels):
        segm = segm.astype(np.uint8) * 255
        bbox = cv2.boundingRect(segm)
        contours, _ = cv2.findContours(segm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        object_id = str(uuid.uuid4())

        s = []
        for contour in contours:
            contour = contour.squeeze(axis=1)
            x = contour[:, 0]
            y = contour[:, 1]

            s.append({"segm": {"x": x.tolist(), "y": y.tolist()}})

        save_key = (
            "background"
            if "floor" in classes[label]
            or "wall" in classes[label]
            or "ceiling" in classes[label]
            else "objects"
        )

        detected_objects[save_key][classes[label]][object_id] = {
            "segms": s,
            "bbox": bbox,
        }
        detected_labels[save_key][classes[label]] += 1

    return {
        "detected_class": dict(detected_labels),
        "results": detected_objects,
    }


def classifier_process(result, label_list):
    detected_class = np.unique([label.split(":")[0] for label in label_list])

    accum_result = {key: [] for key in detected_class}

    for idx, (r, label) in enumerate(zip(result, label_list)):
        l = label.split(":")[0]
        accum_result[l].append(r)
        # results.append({lab: float(conf) for lab, conf in zip(class_label, confidence)})

    results = {}
    for label, conf in accum_result.items():
        accum = np.sum(conf, axis=0) / len(conf)
        accum = accum.tolist()

        results[label] = accum

    return results


def make_bbox_images(detected_objects, img):
    layers = np.zeros(shape=(0, 100, 100, 3), dtype=np.uint8)

    label_list = []

    for i_, (label, objs) in enumerate(detected_objects.items()):
        objs = dict(objs)

        for obj_id, segms in objs.items():
            x, y, w, h = segms["bbox"]
            clipped_result = np.zeros(shape=img.shape, dtype=np.uint8)

            for segm in segms["segms"]:
                s = np.array(
                    list(zip(segm["segm"]["x"], segm["segm"]["y"])), dtype=np.int32
                )
                mask = np.zeros_like(clipped_result)
                mask = cv2.fillPoly(mask, [s], (255, 255, 255))
                clipped_result = cv2.add(clipped_result, cv2.bitwise_and(img, mask))

            clipped_result = clipped_result[y : y + h, x : x + w]
            clipped_result = cv2.resize(clipped_result, (100, 100))
            layers = np.concatenate([layers, clipped_result[np.newaxis]])

            label_list.append(f"{label}:{obj_id}")

    return layers, label_list
