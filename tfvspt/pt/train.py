"""Torch Training."""

import torch
from tqdm.autonotebook import tqdm

from tfvspt.config.config import Config


def train_model(dataloader, model, criterion, optimizer, config: Config,
                device: str) -> dict:

    model.to(device)

    training_logs = {"loss": [], "accuracy": []}
    for epoch in range(config.epochs):

        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        print(f"Epoch {epoch + 1} / {config.epochs}")
        for images, targets in tqdm(dataloader):

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, targets[:, 0])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = torch.argmax(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets[:, 0]).sum().item()

        # Epoch Stats
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = round(correct / total, 4)

        # Collect logs
        training_logs["loss"].append(epoch_loss)
        training_logs["accuracy"].append(epoch_accuracy)

        print(
            f"Epoch {epoch + 1}/{config.epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}"
        )

        print("\n")

    return training_logs
