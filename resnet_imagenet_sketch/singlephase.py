import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--net_size', type=int, choices=[18, 34, 50])
    args = parser.parse_args()
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet'+str(args.net_size), pretrained=True)
