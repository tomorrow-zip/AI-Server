import bentoml
import pickle
import pandas as pd

with open("./checkpoints/umap_model.sav", "rb") as umap_model:
    umap = pickle.load(umap_model)

with open("checkpoints/furniture_vector.pickle", "rb") as f:
    furniture_style = pd.DataFrame(pickle.load(f))

saved_model = bentoml.picklable_model.save_model(
    "recommender_furniture", umap, custom_objects={"furniture_style": furniture_style}
)
print(saved_model)
