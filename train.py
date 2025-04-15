import torch
from dino_loader import load_dino_model
from model import Colorizer

def main():
    dino_model = load_dino_model()
    model = Colorizer(dino_model)

    # Dummy grayscale input for now
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # (1, 3, 224, 224)

if __name__ == "__main__":
    main()