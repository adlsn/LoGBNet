import torch
from monai.inferers import sliding_window_inference
from sLoGNN import sLoGNN
import vmtk
import matplotlib.pyplot as plt


class TestModel:
    def __init__(self, model_path):
        self.model = sLoGNN(1, 2)  # Adjust input and output channels as needed
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def infer(self, input_tensor):
        with torch.no_grad():
            return sliding_window_inference(
                inputs=input_tensor,
                roi_size=(64, 64, 64),  # Adjust region of interest size as needed
                sw_batch_size=1,
                predictor=self.model
            )
