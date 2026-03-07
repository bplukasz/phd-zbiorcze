import os

print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("ERROR importing torch:", e)