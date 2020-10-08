import numpy as np
import matplotlib.pyplot as plt


def idx_to_class(class_to_idx):
    return {v: k for k, v in list(class_to_idx.items())}


def show_imgs(dataset, n_imgs, plot_size=(15, 15), cmap=None):
    """ Show images preasent in the dataset, helpful for initial dataset inspection """
    n_cols = int(np.sqrt(n_imgs))
    n_rows = int(np.ceil(np.sqrt(n_imgs)))
    class_idx = dataset.class_to_idx
    idx_class = idx_to_class(class_idx)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_size)
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        title = f'Class : {idx_class[dataset.targets[i]]}'
        ax.imshow(dataset.data[i], cmap=cmap)
        ax.set_title(title)
    fig.tight_layout()


def show_batch(dataloader):
    """ Show images in one batch along with labels """
    bs = dataloader.batch_size
    num_samples = dataloader.dataset.data.shape[0]
    batches = num_samples // bs
    batch_id = np.random.choice(batches)
    one_batch = list(dataloader)[batch_id]
    batch_imgs, batch_labels = one_batch[0], one_batch[1]
    class_idx = dataloader.dataset.class_to_idx
    idx_class = idx_to_class(class_idx)
    n_rows = n_cols = int(np.sqrt(len(batch_imgs)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    if batch_imgs.shape[1] == 1:
        cmap = 'gray'
    else:
        cmap = None
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        title = f'Class : {idx_class[batch_labels[i].item()]}'
        single_img = np.clip(batch_imgs[i].squeeze().permute(1, 2, 0).numpy(), 0, 1)
        ax.imshow(single_img, cmap=cmap)
        ax.set_title(title)
    fig.tight_layout()

def plot_incorrect_images(img_list, class_idx, cmap=None, plot_size=(15, 15)):
    """ plotting incorrect images """
    n_images = len(img_list)
    n_cols = int(np.sqrt(n_images))
    n_rows = int(np.ceil(np.sqrt(n_images)))
    idx_class = idx_to_class(class_idx)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_size)
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        if i < n_images:
            target_id = img_list[i]['target']
            pred_id = img_list[i]['pred']
            title = f"Target: {idx_class[target_id]} \n  Pred : {idx_class[pred_id]}"
            incorrect_image = np.clip(img_list[i]['img'].permute(1, 2, 0).cpu().numpy() * 0.25 + 0.5, 0, 1)
            ax.imshow(incorrect_image, cmap=cmap)
            ax.set_title(title)
    fig.tight_layout()

def plot_metrics(metric_list, plot_type="Loss"):
    fig, ax = plt.subplots(figsize=(6, 6))
    for metric in metric_list:
        ax.plot(metric['metric'], label=metric['label'])

    ax.set_title(f'Variation of {plot_type.lower()} with epochs', fontsize=14)
    ax.set_ylabel(plot_type, fontsize=10)
    ax.set_xlabel('Number of Epochs', fontsize=10)
    ax.legend()
    fig.tight_layout()