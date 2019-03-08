import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from model import C3D
from data_loader import get_loader

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    transform = None

    # Build data loader
    data_loader = get_loader(root=args.video_root,
                             list_file=args.list_file,
                             transform=transform,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)

    # Build the models
    model = C3D(args.num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, sample in enumerate(data_loader):

            # Set mini-batch dataset
            clip, target = sample
            clip = clip.to(device)
            target = target.to(device)
            # clip = sample['clip'].to(device)
            # target = sample['label'].to(device)

            # Forward, backward and optimize
            outputs = model(clip)
            # preds = torch.max(outputs, 1)[1]
            loss = criterion(outputs, target)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'c3d-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path and record
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--video_root', type=str, default='D:/Workspace/dataset/UCF-101/', help='directory for videos')
    parser.add_argument('--list_file', type=str, default='ucfTrainTestlist/trainlist02.txt',
                        help='directory for videos list file')
    parser.add_argument('--log_step', type=int, default=10, help='steps for printing log info')
    parser.add_argument('--save_step', type=int, default=1000, help='steps for saving trained models')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
