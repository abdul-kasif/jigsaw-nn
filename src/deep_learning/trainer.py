import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one complete epoch.

    Returns:
        Average training loss across all samples.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # Move batch tensors to the selected device.
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)

        # Clear gradients from the previous update.
        optimizer.zero_grad()

        # Forward pass.
        logits = model(input_ids, lengths)

        # Calculate loss.
        loss = criterion(logits, labels)

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Accumulate loss weighted by batch size.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model without updating its parameters.

    Returns:
        Average validation loss across all samples.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Disable gradient calculations during evaluation.
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"]
            labels = batch["labels"].to(device)

            # Forward pass.
            logits = model(input_ids, lengths)

            # Calculate loss.
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss


def predict(model, dataloader, device):
    """
    Generate probabilities and collect ground-truth labels.

    Returns:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
    """

    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"]
            labels = batch["labels"]

            logits = model(input_ids, lengths)

            # Convert logits into probabilities.
            probabilities = torch.sigmoid(logits)

            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())

    y_true = torch.cat(all_labels, dim=0)
    y_prob = torch.cat(all_probabilities, dim=0)

    return y_true, y_prob
