import pandas as pd
import pickle

from umap import UMAP

import scipy.spatial.distance as distance


class RecommenderFurniture:
    def __init__(self):
        self.umap = pickle.load((open("checkpoints/umap_model.sav", "rb")))

        with open("checkpoints/furniture_vector.pickle", "rb") as f:
            self.furniture_style = pd.DataFrame(pickle.load(f))

        self.cate_num = {1: 3, 2: 1, 3: 4}  # 침대, 소파, 옷장

    def calculate_similarity(self, vector1, vector2):
        return distance.euclidean(vector1[0], vector2[0])

    def get_vector(self, values):
        vectors = self.umap.transform(values)
        return vectors

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
