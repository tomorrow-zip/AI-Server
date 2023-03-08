import bentoml


detection_model = bentoml.pytorch.get("detection_and_segmentation:latest")
classifier_model = bentoml.keras.get("style-classifier:latest")
recommend_model = bentoml.picklable_model.get("recommender_furniture:latest")
