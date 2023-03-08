import bentoml
import scipy.spatial.distance as distance

from models import recommend_model


class RecommenderRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = "cpu"
    SUPPORTS_CPU_MULTI_THREADING = False

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
