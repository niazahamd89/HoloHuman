from camera import Camera
from video import generate_video_results
from humvis import HumVis



##For Testing the model###

import torch
import cv2
import os

# Load the pre-trained model
model = HumVis(backbone_name='hrnet32')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)

################################################################

