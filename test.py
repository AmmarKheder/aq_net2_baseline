import torch
print("GPU disponible :", torch.cuda.is_available())
print("Nombre de GPU  :", torch.cuda.device_count())
