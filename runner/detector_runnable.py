import torch
import bentoml

from mmdet.apis import inference_detector

from models import detection_model


class DetectorRunnable(bentoml.Runnable):
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
