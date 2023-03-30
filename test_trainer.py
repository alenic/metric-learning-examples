from mlelib import *
import timm
import pytorch_metric_learning.losses as losses
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib

num_classes = 18
dataset_root = "/media/node_ale/DATA/datasets/celebrity-faces-dataset"

def visualize_features(ax, features, label_ids, label_names, title=""):
    cmap = matplotlib.cm.get_cmap('tab20')
    for cid in np.unique(label_ids):
        select = label_ids == cid
        ax.scatter(features[select, 0],
                    features[select,1],
                    s=8,
                    color=cmap(cid),
                    label=label_names[cid])

    ax.legend()
    ax.title.set_text(title)



if __name__ == "__main__":
    train_tr = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_tr = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    with torch.no_grad():
        n_features = model(torch.rand((1,3,224,224))).shape[-1]
    model = torch.nn.Sequential(model, torch.nn.Linear(n_features, 2))
    dataset = datasets.ImageFolder(dataset_root, transform=train_tr)

    label_names = dataset.classes
    print(label_names)

    train_idx, val_idx = train_val_split_indices(len(dataset), val_perc=0.25, random_state=42)
    train_dataset = Subset(dataset, indices=train_idx)
    val_dataset = Subset(dataset, indices=val_idx)
    val_dataset.dataset.transform = val_tr

    #loss_func = losses.LargeMarginSoftmaxLoss(num_classes=num_classes, embedding_size=2)
    #loss_func = losses.TripletMarginLoss()
    loss_func = losses.CosFaceLoss(num_classes=num_classes, embedding_size=2, scale=12, margin=1)

    trainer = Trainer(model, loss_func=loss_func, lr=1e-4, n_epochs=20)

    trainer.fit(train_dataset, val_dataset=val_dataset, early_stopping_patience=5)

    train_embeddings, train_labels = trainer.get_embeddings(train_dataset, return_labels=True, to_numpy=True)
    val_embeddings, val_labels = trainer.get_embeddings(val_dataset, return_labels=True, to_numpy=True)

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    visualize_features(ax[0], train_embeddings, train_labels, label_names)
    visualize_features(ax[1], val_embeddings, val_labels, label_names)
    
    plt.show()