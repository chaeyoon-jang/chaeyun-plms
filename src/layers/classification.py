import torch
import torch.nn as nn
import torch.nn.functional as F 
from layers import MultiTaskClassifier

class ClassificationTask(MultiTaskClassifier):
    def __init__(self, task_def):
        super().__init__(task_def)

    @classmethod
    def 