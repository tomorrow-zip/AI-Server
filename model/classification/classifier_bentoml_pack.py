from keras import Model

import tensorflow as tf
import bentoml


def load_similarity_model(model_path, fc_layer):
    _model = tf.keras.models.load_model(model_path)
    return Model(_model.input, _model.output)


model = load_similarity_model("checkpoints", "dense_5")

model.summary()

labels = [
    "asian",
    "beach",
    "beds",
    "chairs",
    "contemporary",
    "craftsman",
    "dressers",
    "eclectic",
    "farmhouse",
    "industrial",
    "lamps",
    "mediterranean",
    "midcentury",
    "modern",
    "rustic",
    "scandinavian",
    "sofas",
    "tables",
    "traditional",
    "transitional",
    "tropical",
]

save_model = bentoml.keras.save_model(
    "style-classifier", model=model, custom_objects={"classes": labels}
)
