import torch
import torch.nn.functional as F

def load_openclip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    # preprocess to PIL->tensor; my jednak już mamy tensory [-1,1] z datasetu
    # więc robimy własny preprocess w extract.
    return model

@torch.no_grad()
def clip_image_features(model, x):
    """
    x: Bx3xHxW in [-1,1]
    output: Bxd normalized
    """
    # CLIP expects 224x224, input in [0,1] then normalized internally? open_clip model expects normalized input.
    # Najprościej: skorzystaj z open_clip's preprocess stats:
    # mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    x = (x + 1) / 2
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
    x = (x - mean) / std
    f = model.encode_image(x)
    f = F.normalize(f.float(), dim=1)
    return f

