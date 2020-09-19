from tqdm import tqdm
import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, reg, lambda1, lambda2, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate Loss
        regularization_loss1 = 0
        regularization_loss2 = 0
        if (reg == 'None'):  # No Regularization
            loss = F.nll_loss(y_pred, target)
            train_losses.append(loss)

        elif (reg == 'L1'):  # Loss with L1
            loss = F.nll_loss(y_pred, target)
            for param in model.parameters():
                regularization_loss1 += torch.norm(param, 1)
            loss += (lambda1 * regularization_loss1)
            train_losses.append(loss)

        elif (reg == 'L2'):  # Loss with L2
            loss = F.nll_loss(y_pred, target)
            for param in model.parameters():
                regularization_loss2 += torch.norm(param, 2)
            loss += (lambda2 * regularization_loss2)
            train_losses.append(loss)

        elif (reg == 'L1L2'):  # Loss with L1 and L2
            loss = F.nll_loss(y_pred, target)
            for param in model.parameters():
                regularization_loss1 += torch.norm(param, 1)
                regularization_loss2 += torch.norm(param, 2)
            loss += (lambda1 * regularization_loss1 + lambda2 * regularization_loss2)
            train_losses.append(loss)

        elif (reg == 'GBN'):  # Loss with GBN
            loss = F.nll_loss(y_pred, target)
            train_losses.append(loss)

        else:  # Loss with GBN, L1 and L2
            loss = F.nll_loss(y_pred, target)
            for param in model.parameters():
                regularization_loss1 += torch.norm(param, 1)
                regularization_loss2 += torch.norm(param, 2)
            loss += (lambda1 * regularization_loss1 + lambda2 * regularization_loss2)
            train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item():0.3f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_acc.append(100. * correct / processed)


def test(model, device, test_loader, test_losses, test_acc, misclassified_images, misclassified=False):
    model.eval()
    test_loss = 0
    correct = 0
    num_misclassified = 0
    with torch.no_grad():
        for data, target in test_loader:
            # if misclassified:
            #   batch_images = data
            #   batch_target = target
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if misclassified:
                if num_misclassified <= 25:
                    incorrect_id = ~pred.eq(target.view_as(pred))
                    # incorrect_images =  batch_images[incorrect_id]
                    # incorrect_preds = pred[incorrect_id]
                    # incorrect_target = batch_target[incorrect_id]
                    if incorrect_id.sum().item() != 0:
                        num_misclassified += incorrect_id.sum().item()
                        misclassified_images.append(
                            (data[incorrect_id], pred[incorrect_id], target.view_as(pred)[incorrect_id]))

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
