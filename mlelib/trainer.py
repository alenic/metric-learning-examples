import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
import numpy as np
import copy
import os

class Trainer:
    def __init__(self,
                 model,
                 loss_func,
                 output_dir="outputs",
                 miner_func=None,
                 optim="adam",
                 lr=1e-3,
                 weight_decay=0,
                 lr_sched="cosannealing",
                 batch_size=32,
                 n_epochs=40,
                 num_workers=None,
                 device=None
                 ):
    
        self.model = model
        self.loss_func = loss_func
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.miner_func = miner_func

        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = mp.cpu_count()

        parameters =list(model.parameters()) + list(loss_func.parameters())

        if optim == "adam":
            self.optim = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

        if lr_sched == "cosannealing":
            self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=n_epochs, eta_min=0.01*lr)
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        self.loss_func.to(self.device)
    
    def save_model(self, output_dir=None):
        if output_dir:
            outdir = output_dir
        else:
            outdir = self.output_dir
        
        os.makedirs(outdir, exist_ok=True)
        torch.save(self.best_dict, os.path.join(outdir, "best_model.pth"))

    def get_embeddings(self, dataset, model=None, return_labels=False, to_numpy=False):
        val_loader = DataLoader(dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size,
                                shuffle=False)
        emb_list = []
        if model is None:
            model = self.model
        
        model.eval()
        model.to(self.device)
        label_list = []
        for iter, (image_batch, label_batch) in enumerate(val_loader):
            image_batch = image_batch.to(self.device)
            label_batch = label_batch.to(self.device)

            label_list.append(label_batch.cpu().numpy())

            with torch.no_grad():
                embeddings = model(image_batch)

            if to_numpy:
                emb_list.append(embeddings.cpu().numpy())
            else:
                emb_list.append(embeddings)
        
        if to_numpy:
            emb = np.vstack(emb_list)
            labels = np.concatenate(label_list)
        else:
            emb = torch.stack(emb_list)
            labels = torch.concat(label_list)
        
        if return_labels:
            return emb, labels
        
        return emb


    def fit(self, train_dataset, val_dataset=None, early_stopping_patience=5):
        train_loader = DataLoader(train_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size,
                                shuffle=True,
                                drop_last=True)
        if val_dataset:
            val_loader = DataLoader(val_dataset,
                                    num_workers=self.num_workers,
                                    batch_size=self.batch_size,
                                    shuffle=False)
            eval = True
            best_mean_loss = np.inf
            early_stoping_counter = 0
        else:
            eval = False
        
        for epoch in range(self.n_epochs):
            self.model.train()
            loss_list = []
            for iter, (image_batch, label_batch) in enumerate(train_loader):
                self.optim.zero_grad()

                image_batch = image_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                embeddings = self.model(image_batch)

                if self.miner_func:
                    miner_output = self.miner_func(embeddings, label_batch)
                    loss = self.loss_func(embeddings, label_batch, miner_output)
                else:
                    loss = self.loss_func(embeddings, label_batch)
                
                loss_list.append(loss.item())

                loss.backward()
                self.optim.step()
            
            mean_train_loss = np.mean(loss_list)
            print(f"epoch {epoch} Train Loss {mean_train_loss}")

            self.lr_sched.step()

            if eval:
                loss_list = []
                self.model.eval()
                for iter, (image_batch, label_batch) in enumerate(val_loader):
                    image_batch = image_batch.to(self.device)
                    label_batch = label_batch.to(self.device)

                    with torch.no_grad():
                        embeddings = self.model(image_batch)

                    loss = self.loss_func(embeddings, label_batch)
                    loss_list.append(loss.item())
                
                mean_val_loss = np.mean(loss_list)
                print(f"epoch {epoch} Val Loss {mean_val_loss}")

                if mean_val_loss < best_mean_loss:
                    best_mean_loss = mean_val_loss
                    self.best_dict = {
                        "epoch": epoch,
                        "model": copy.deepcopy(self.model.state_dict()),
                        "loss": copy.deepcopy(self.loss_func.state_dict()),
                        "optim": copy.deepcopy(self.optim.state_dict()),
                        "lr_sched": copy.deepcopy(self.lr_sched.state_dict())
                    }
                    self.save_model(self.output_dir)

                    early_stoping_counter = 0
                else:
                    early_stoping_counter += 1
                
                if early_stoping_counter == early_stopping_patience:
                    return
                    




