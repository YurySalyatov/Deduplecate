import torch
print(torch.cuda.is_available())  # Должно вернуть True
print(torch.version.cuda)