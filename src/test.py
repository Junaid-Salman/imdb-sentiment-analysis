import numpy as np
import torch
import sklearn

print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Scikit-learn:", sklearn.__version__)

print("CUDA Available:", torch.cuda.is_available())
