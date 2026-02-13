import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionFeatureExtractor(nn.Module):
    """
    Zwraca pool3 2048-d features jak w FID.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        # Nowsza wersja torchvision wymaga aux_logits=True z pretrenowanymi wagami
        m = inception_v3(weights=weights, transform_input=False, aux_logits=True)
        m.eval()
        # "odcinamy" klasyfikator, bierzemy do Mixed_7c + pooling
        self.m = m.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # x: [-1,1], Bx3xHxW -> 299
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Inception expects normalized to ImageNet stats implicitly in weights transforms,
        # ale w praktyce FID implementacje robią swoje; tutaj trzymamy prosty wariant.
        # Jeśli chcesz zgodność z clean-fid, potem dopasujemy preprocessing.
        m = self.m
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.flatten(1)
        return x

