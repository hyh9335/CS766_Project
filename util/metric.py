import torch
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):  #input=GT, outputs=predicted
        inputs = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(inputs.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return 1, 1

        true_positive = ((outputs == inputs) * inputs).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall