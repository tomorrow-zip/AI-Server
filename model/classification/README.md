# About Checkpoint

---

We referenced the [ObjectDetectionProject-IKEAFurnituresRecommender](https://github.com/sophiachann/ObjectDetectionProject-IKEAFurnituresRecommender/tree/main/model) model to classify furniture styles.
So [train.py](./style_classifier_train.py) is written with similar content.

We use [Bonn Furniture Dataset](https://cvml.comp.nus.edu.sg/furniture/index.html) and `df_train.csv`, `df_valid.csv`, `df_test.csv` is written to `image_path(filename)` and `columns` to one-hot vectorized.

The columns are as follows

```text
[
    "Asian",
    "Beach",
    "Contemporary",
    "Craftsman",
    "Eclectic",
    "Farmhouse",
    "Industrial",
    "Mediterranean",
    "Midcentury",
    "Modern",
    "Rustic",
    "Scandinavian",
    "Southwestern",
    "Traditional",
    "Transitional",
    "Tropical",
    "Victorian",
]
```

You can create a CSV file containing the file path and label name yourself, or if possible, write a custom data loader code by referring to the model structure only.