from mlelib import *
import timm
import pytorch_metric_learning.losses as losses
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T


if __name__ == "__main__":
    train_tr = T.Compose([T.Resize((224,224), T.RandomHorizontalFlip(), T.ToTensor())])

    model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    model = torch.nn.Sequential(model, torch.nn.Linear(512, 2))
    train_dataset = datasets.ImageFolder("C:/Users/xxale/Documents/datasets/taiwanese_food_101/train", transform=train_tr)

    loss_func = losses.LargeMarginSoftmaxLoss(num_classes=10, embedding_size=2)

    trainer = Trainer(model, loss_func=loss_func)

    trainer.fit(train_dataset)