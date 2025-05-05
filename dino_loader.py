import torch

def load_dino_model():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

    model.eval()  
    model.token_dim = 768 
    return model
