# train.py
import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_loader.dataset.reset_fill_counter()

    for batch in tqdm(train_loader, desc="Training", leave=False):
        if batch is None:
            continue
        eegs, labels, _, _ = batch
        eegs = eegs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(eegs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_loss = train_loss / len(train_loader)
    acc = 100.0 * train_correct / train_total

    filled = train_loader.dataset.get_filled_count()
    print(f"Epoch fill stats: {filled} samples had NaN/Inf were forward-filled")
    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            if batch is None:
                continue
            eegs, labels, _, _ = batch
            eegs = eegs.to(device)
            labels = labels.to(device)

            outputs = model(eegs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    acc = 100.0 * val_correct / val_total
    return avg_loss, acc


def test(model, test_loader, device):
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None:
                continue
            eegs, labels, _, _ = batch
            eegs = eegs.to(device)
            labels = labels.to(device)

            outputs = model(eegs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    acc = 100.0 * test_correct / test_total
    return acc
