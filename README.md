# Tomorrow Zip AI Server

---

장면 분할, 분위기 검출, 가구 추천 모델 서빙 서버


## Project Environment

- python 3.8
- bentoml@1.0.15
- torch@1.10.1+cu113

- cuda@11.2
- nvidia@460.106


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

```shell
tomorrow-zip-ai-api
--------------------------------------
|-- README.md
|-- apis
|   `-- openapi.yaml
|-- bento.yaml
|-- env
|   |-- docker
|   |   |-- Dockerfile
|   |   |-- Dockerfile.template
|   |   `-- entrypoint.sh
|   `-- python
|       |-- install.sh
|       |-- requirements.lock.txt
|       |-- requirements.txt
|       `-- version.txt
|-- models
|   |-- detection_and_segmentation
|   |   |-- 2cjttif3jgfnd32z
|   |   |   |-- custom_objects.pkl
|   |   |   |-- model.yaml
|   |   |   `-- saved_model.pt
|   |   `-- latest
|   |-- recommender_furniture
|   |   |-- h6anadf3yk42j4op
|   |   |   |-- custom_objects.pkl
|   |   |   |-- model.yaml
|   |   |   `-- saved_model.pkl
|   |   `-- latest
|   `-- style-classifier
|       |-- latest
|       `-- sd64eef3jgfnd32z
|           |-- assets
|           |-- custom_objects.pkl
|           |-- keras_metadata.pb
|           |-- model.yaml
|           |-- saved_model.pb
|           `-- variables
|               |-- variables.data-00000-of-00001
|               `-- variables.index
`-- src
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


Make for Environments
```shell
pip install -r requirements.txt
```


Let's build a system through BentoML
```shell
bentoml build
```

and RUN!
```shell
bentoml serve 
```

<br>

### Project containerize

after bentoml build, if you already build a project, you don't have to do it.
```shell
bentoml containerize tomorrow-zip-ai-api:latest -t tomorrow-zip-ai-api 
```

Docker RUN
```shell
docker run -it --rm --name tomorrow-zip-ai-serving --gpus all -p 3000:3000 -p 3001:3001 tomorrow-zip-ai-api:latest serve
```


## 고려한 점

- N개의 모델을 한정된 자원에서 Serving이 가능하게
- 서로 다른 형식의 Model을 하나의 포맷으로 관리