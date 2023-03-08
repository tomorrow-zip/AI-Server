# models
from models import detection_model, recommend_model, classifier_model

# Runners
from runner.detector_runnable import DetectorRunnable
from runner.recommender_runnable import RecommenderRunnable

# utils
import bentoml
from bentoml.io import Image, JSON, Multipart

from flask import Flask

import numpy as np

# processing
from processing import detect_process, classifier_process, make_bbox_images


detector_runner = bentoml.Runner(DetectorRunnable, models=[detection_model])
recommend_runner = bentoml.Runner(RecommenderRunnable, models=[recommend_model])
classifier_runner = classifier_model.to_runner()


# service 선언
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


@svc.api(
    input=Multipart(image=Image(), detected_objects=JSON()),
    output=JSON(),
)
def classifier(image, detected_objects):
    image = np.array(image, dtype=np.uint8)

    layers, label_list = make_bbox_images(detected_objects, image)
    result = classifier_runner.run(layers)

    return classifier_process(result, label_list)


@svc.api(input=JSON(), output=JSON())
def recommend(classified):
    sim = {}

    for label, styles in classified.items():
        sim[label] = list(recommend_runner.run(styles).index)

    return {"result": sim}


@svc.api(input=Image(), output=JSON())
def unified(image):
    res = detect(image)
    clas_pred = classifier(image, res["results"]["objects"])
    recommendation = recommend(clas_pred)

    return {
        "detected_class": res["detected_class"],
        "detected_objects": res["results"]["objects"],
        "style": clas_pred,
        "recommendation": recommendation,
    }


"""
Flask 객체 생성
"""
app = Flask(__name__)
svc.mount_wsgi_app(app)


@app.route("/metadata")
def metadata():
    return {
        "detection-model-name": detection_model.tag.name,
        "detection-model-version": detection_model.tag.version,
        "detection-model-classes": len(detection_model.custom_objects["classes"]),
        "style-classifier-model-name": classifier_model.tag.name,
        "style-classifier-version": classifier_model.tag.version,
        "style-classifier-classes": len(classifier_model.custom_objects["classes"]),
        "recommender-model-name": recommend_model.tag.name,
        "recommender-model-version": recommend_model.tag.version,
        "recommender-model-classes": len(
            recommend_model.custom_objects["furniture_style"]
        ),
    }
