# inference.py
import torch
from base import *

# Load your model
model.restore('PINN/model_trial-10000.pt', verbose=1)

def pinn_inference(input_value):
    # Get prediction
    prediction = model.predict(torch.tensor([input_value]))
    return prediction.item()