from mmdet.apis import inference_detector
from bentoml.io import Image, JSON, NumpyNdarray
from flask import Flask

from processing import *

from umap import UMAP

import bentoml
import numpy as np
import cv2
import pandas as pd
import pickle
import torch

import uuid
import scipy.spatial.distance as distance


detection_model = bentoml.pytorch.get("detection_and_segmentation:latest")
classifier_model = bentoml.keras.get("style-classifier:latest")
recommend_model = bentoml.picklable_model.get("recommender_furniture:latest")


"""
Custom Runner 정의
"""


class MMDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/cuda", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = bentoml.pytorch.load_model(detection_model)
        if torch.cuda.is_available():
            self.model.cuda()
        # elif torch.backends.mps.is_available():
        #     self.model = self.model.to("mps")

        self.classes = self.model.CLASSES

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def detect(self, input_tensor):
        return inference_detector(self.model, input_tensor)


class RecommenderFurnitureRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = "cpu"

    def __init__(self):
        self.model = bentoml.picklable_model.load_model(recommend_model)
        self.furniture_style = recommend_model.custom_objects["furniture_style"]

        self.cate_num = {1: 3, 2: 1, 3: 4}  # 침대, 소파, 옷장

    def calculate_similarity(self, vector1, vector2):
        return distance.euclidean(vector1[0], vector2[0])

    def get_vector(self, values):
        vectors = self.model.transform(values)
        return vectors

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def compare_similarity(self, arr, cate=0):
        # image_arr 를 벡터화시킴
        vector_obj = self.get_vector([arr])
        # 유사도를 구함
        self.furniture_style["similarity"] = self.furniture_style["vector"].apply(
            lambda x: self.calculate_similarity(x, vector_obj)
        )

        furniture_sort = self.furniture_style.sort_values(
            by=["similarity"], ascending=True
        )
        if cate == 0:
            return furniture_sort.head(10)
        else:
            return furniture_sort[furniture_sort["cate"] == self.cate_num[cate]].head(
                10
            )


"""
Runner 선언
"""
detector_runner = bentoml.Runner(MMDetectionRunnable, models=[detection_model])
recommend_runner = bentoml.Runner(
    RecommenderFurnitureRunnable, models=[recommend_model]
)
classifier_runner = classifier_model.to_runner()

svc = bentoml.Service(
    "tomorrow-zip-ai-api",
    runners=[detector_runner, classifier_runner, recommend_runner],
)


"""
API 구현
"""


@svc.api(input=Image(), output=JSON())
def detect(input_img):
    input_img = np.array(input_img, dtype=np.uint8)

    result = detector_runner.run(input_img)
    classes = detection_model.custom_objects["classes"]

    return detect_process(result, classes)


@svc.api(input=NumpyNdarray(), output=JSON())
def classifier(input_images):
    input_img = np.array(input_images, dtype=np.uint8)

    if input_img.shape[0] != 100 and input_img.shape[1] != 100:
        input_img = cv2.resize(input_img, (100, 100))

    input_img = input_img[np.newaxis]
    result_classfied = classifier_runner.run(input_img)
    return classifier_process(result_classfied)


@svc.api(input=JSON(), output=JSON())
def recommend(classified_categories):
    sim = {}
    for label, styles in classified_categories.items():
        sim[label] = list(recommend_runner.run(styles).index)

    return {"result": sim}


@svc.api(input=Image(), output=JSON())
def unified(input_img):
    classes = detection_model.custom_objects["classes"]
    res = detect_process(input_img, classes)
    img = np.array(input_img, dtype=np.uint8)

    detected_classes = res["detected_class"]
    detected_objects = res["results"]["objects"]

    layers, label_list = make_bbox_images(detected_objects, img)
    clas_pred = classifier_process(layers, label_list)
    recommended_objects = find_similarity(clas_pred)

    return {
        "state": "success",
        "detected_class": detected_classes,
        "detected_object_location": detected_objects,
        "style": clas_pred,
        "recom": recommended_objects,
    }


"""
Flask 객체 생성
"""
app = Flask(__name__)
svc.mount_wsgi_app(app)


@app.route("/metadata")
def metadata():
    return {
        "name": detection_model.tag.name,
        "version": detection_model.tag.version,
        "classes": len(detection_model.custom_objects["classes"]),
    }
