import torch
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

def load_dino_model():
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = vit_b_16(weights=weights)
    model.eval()
    model.token_dim = model.heads.head.in_features
    return model
