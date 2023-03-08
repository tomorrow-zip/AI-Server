# Tomorrow Zip AI Server

---

장면 분할, 분위기 검출, 가구 추천 모델 서빙 서버


## Project Environment

- python 3.8
- bentoml@1.0.15
- torch@1.10.1+cu113

- cuda@11.2
- nvidia@460.100


## Project Structure
```shell
tomorrow-zip-ai-server
|-- Dockerfile.template
|-- README.md
|-- bentofile.yaml
|-- model
|   |-- __init__.py
|   |-- classification
|   |   |-- checkpoints
|   |   |   |-- saved_model.pb
|   |   |   `-- variables
|   |   |       |-- variables.data-00000-of-00001
|   |   |       `-- variables.index
|   |   |-- classifier_bentoml_pack.py
|   |   `-- style_classifier_train.py
|   |-- detection
|   |   |-- checkpoints
|   |   |   |-- detector.config.py
|   |   |   `-- mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth
|   |   `-- mask2former_bentoml_pack.py
|   `-- recommendation
|       |-- checkpoints
|       |   |-- furniture_vector.pickle
|       |   `-- umap_model.sav
|       |-- recommend-test.ipynb
|       |-- recommender_bentoml_pack.py
|       `-- recommender_furniture.py
|-- models.py
|-- processing.py
|-- runner
|   |-- __init__.py
|   |-- detector_runnable.py
|   `-- recommender_runnable.py
`-- service.py
```

## Setting

---

### Project build & run

```shell
bentoml build
bentoml serve 
```

### Project containerize
```shell
# after bentoml build ...
# if you already build a project, you don't have to do it

bentoml containerize tomorrow-zip-ai-api:latest -t tomorrow-zip-ai-api 
docker run -it --rm --name tomorrow-zip-ai-serving --gpus all -p 3000:3000 -p 3001:3001 tomorrow-zip-ai-api:latest serve
```


## 고려한 점

- N개의 모델을 한정된 자원에서 Serving이 가능하게
- 서로 다른 형식의 Model을 하나의 포맷으로 관리