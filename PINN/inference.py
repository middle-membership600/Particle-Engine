import sys
import torch
from base import *

# Load your model
model.restore('PINN/model_trial-10000.pt', verbose=1)

# Get input from command line
input_value = float(sys.argv[1])

# Get prediction
prediction = model.predict(torch.tensor([input_value]))

# Write the prediction to a temporary file
with open("/tmp/model_output.txt", "w") as file:
    file.write(str(prediction.item()))
