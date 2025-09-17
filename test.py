import torch

if __name__ == "__main__":
    w = torch.zeros((5, 6))
    x = torch.zeros((3, 4, 5))
    print(x.transpose(0, 0))
