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


def make_bbox_images(detected_objects, img, resize_width=100, resize_height=100):
    if len(img.shape) == 2:  # 1채널 이미지 처리
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    layers = np.zeros(shape=(0, resize_height, resize_width, 3), dtype=np.uint8)
    labels = []

    for label, objs in detected_objects.items():
        for obj_id, segms_data in objs.items():
            segms = segms_data["segms"]
            x, y, w, h = segms_data["bbox"]
            clipped_result = np.zeros(shape=img.shape, dtype=np.uint8)

            mask = None
            for segm in segms:
                contour_points = np.array(
                    list(zip(segm["segm"]["x"], segm["segm"]["y"])), dtype=np.int32
                )
                mask = np.zeros_like(img)
                mask = cv2.fillPoly(mask, [contour_points], (255, 255, 255))
                clipped_result_portion = cv2.bitwise_and(img, mask)
                clipped_result = cv2.add(clipped_result, clipped_result_portion)

            inverse_mask = cv2.bitwise_not(mask)
            white_bg = np.full(img.shape, 255, dtype=np.uint8)
            white_bg_portion = cv2.bitwise_and(white_bg, inverse_mask)

            clipped_result = cv2.add(clipped_result, white_bg_portion)
            clipped_result = clipped_result[y : y + h, x : x + w]
            clipped_result = cv2.cvtColor(clipped_result, cv2.COLOR_RGB2BGR)

            # 비율을 유지한 상태로 resize
            aspect_ratio = float(w) / float(h)
            new_width = resize_width
            new_height = int(resize_width / aspect_ratio)

            if new_height > resize_height:
                new_width = int(resize_height * aspect_ratio)
                new_height = resize_height

            resized_img = cv2.resize(clipped_result, (new_width, new_height))
            top_pad = (resize_height - new_height) // 2
            bottom_pad = resize_height - new_height - top_pad
            left_pad = (resize_width - new_width) // 2
            right_pad = resize_width - new_width - left_pad
            resized_img = cv2.copyMakeBorder(
                resized_img,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )

            layers = np.concatenate([layers, resized_img[np.newaxis]])
            labels.append(f"{label}:{obj_id}")

    return layers, labels
