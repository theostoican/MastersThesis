import numpy as np
import torch

from utils.models import TeacherNetwork

def construct_toy_dataset():
    data = []
    for y in np.arange(-5, 5.1, .25):
        for x in np.arange(-5, 5.1, .25):
            data.append([x, y])

    teacher_model = TeacherNetwork()
    labels = teacher_model(torch.DoubleTensor(data)).detach().numpy()
    
    return data, labels

