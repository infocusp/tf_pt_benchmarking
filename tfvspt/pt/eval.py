"""Torch Evaluation."""

import torch
from tqdm.autonotebook import tqdm


def eval_model(dataloader, model, criterion, device) -> dict:

    model.to(device)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader):

            images, targets = images.to(device), targets.to(device)

            outputs = model(images)

            loss = criterion(outputs, targets[:, 0])

            val_loss += loss.item()

            predicted = torch.argmax(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets[:, 0]).sum().item()

    val_loss /= len(dataloader)
    val_accuracy = round(correct / total, 4)

    return {"loss": val_loss, "accuracy": val_accuracy}
