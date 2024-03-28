import os

from box import Box
from tqdm import tqdm
from time import sleep
from typing import List, Dict
from src.utils.logger import Logger
from src.utils.ConfigManager import ConfigManager
from src.utils.early_stopping import EarlyStopper
from src.utils.utils import init_loss, init_metrics, init_lr_scheduler, init_model_optimizer_start_epoch


import torch
import torcheval

from torch.nn.functional import sigmoid, softmax
from torch.utils.data import DataLoader

from torchvision.transforms import v2 


class Trainer:
    __options: Box
    __train_log_path: str
    __eval_log_path: str
    __checkpoint_path: str
    __device: str
    __best_val_loss: float

    __train_loader: DataLoader
    __validation_loader: DataLoader

    __early_stopper: EarlyStopper
    __logger: Logger

    __loss: torch.nn.Module
    __optimizer: torch.optim.Optimizer
    __lr_schedulers: torch.optim.lr_scheduler.LRScheduler
    __start_epoch: int
    __model: torch.nn.Module


    def __init__(self,
                 configManager: ConfigManager,
                 train_loader: DataLoader,
                 val_loader: DataLoader
                 ):
        self.__config_manager: ConfigManager = configManager
        self.__device: str = "cuda" if self.__config_manager.CUDA else "cpu"
        self.__train_loader: DataLoader = train_loader
        self.__validation_loader: DataLoader = val_loader


        self.__early_stopper: EarlyStopper = EarlyStopper(self.__best_val_loss, **self.__options.SOLVER.EARLY_STOPPING)
        self.__best_val_loss: float = self.__get_best_val_loss()

        self.__logger: Logger = Logger()

        self.__loss = init_loss(name=self.__options.SOLVER.LOSS.NAME, args=self.__options.SOLVER.LOSS.ARGS)
        self.__start_epoch, self.__model, self.__optimizer = init_model_optimizer_start_epoch(device=self.__device,
                                                                                              checkpoint_load=self.__options.CHECKPOINT.LOAD,
                                                                                              checkpoint_path=checkpoint_path,
                                                                                              resume_name=self.__options.CHECKPOINT.RESUME_NAME,
                                                                                              optimizer_name=self.__options.SOLVER.OPTIMIZER.NAME,
                                                                                              optimizer_args=self.__options.SOLVER.OPTIMIZER.ARGS,
                                                                                              model_base=self.__options.SOLVER.MODEL.BASE,
                                                                                              model_name=self.__options.SOLVER.MODEL.NAME,
                                                                                              model_args=self.__options.SOLVER.MODEL.ARGS
                                                                                              )
        self.__lr_schedulers: torch.optim.lr_scheduler.LRScheduler = init_lr_scheduler(
            name=self.__options.SOLVER.LR_SCHEDULER.NAME, args=self.__options.SOLVER.LR_SCHEDULER.ARGS,
            optimizer=self.__optimizer)


    @classmethod
    def __init_subclass__(cls):
        """Check indispensable args when instantiate Trainer"""
        required_class_variables = [
            "__options", "__train_log_path", "__eval_log_path", "__checkpoint_path", "__train_loader", "__val_loader"
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )

    # Setter & Getter
    @property
    def model(self):
        return self.__model

    # Public methods
    def train(self, sleep_time: int = None, metric_in_train: bool = False) -> None:
        """
        sleep_time: temporarily cease the training process
        metric_in_train: compute metrics during training phase or not
        """
        print("Start training model ...")

        for epoch in range(self.__start_epoch, self.__start_epoch + self.__options.EPOCH.EPOCHS):
            print("Epoch:", epoch)

            for phase, dataset_loader, log_path in zip(("train", "eval"),
                                                       (self.__train_loader, self.__validation_loader),
                                                       (self.__train_log_path, self.__eval_log_path)):
                # Preliminary setups
                self.__model.train() if phase == "train" else self.__model.eval()
                metrics: List[torcheval.metrics.Metric] = init_metrics(name_lst=self.__options.METRICS.NAME_LIST,
                                                                       args=self.__options.METRICS.ARGS,
                                                                       device=self.__device) if metric_in_train else None

                # Epoch running
                run_epoch_result: Dict = self.__run_epoch(phase=phase, epoch=epoch, dataset_loader=dataset_loader,
                                                          metrics=metrics)

                # Logging
                self.__logger.write(log_path, {**{"epoch": epoch}, **run_epoch_result})

                if phase == "eval":
                    # Save checkpoint
                    if self.__options.CHECKPOINT.SAVE:
                        self.__save_checkpoint(epoch=epoch, val_loss=run_epoch_result["loss"],
                                               save_all=self.__options.CHECKPOINT.SAVE_ALL,
                                               obj={"epoch": epoch, "val_loss": run_epoch_result["loss"],
                                                    "model_state_dict": self.__model.state_dict(),
                                                    "optimizer_state_dict": self.__optimizer.state_dict()
                                                    }
                                               )
                    print("Early stopper", self.__early_stopper.min_val_loss, self.__early_stopper.counter)
                    # Early stopping checking
                    if self.__options.MISC.APPLY_EARLY_STOPPING:
                        if self.__early_stopper.check(val_loss=run_epoch_result["loss"]):
                            exit()

                # Stop program in the meantime
                if sleep_time is not None:
                    sleep(sleep_time)
        return None

    # Private methods
    def __run_epoch(self, phase: str, epoch: int, dataset_loader: DataLoader,
                    metrics: List[torcheval.metrics.Metric] = None) -> Dict:
        """
        phase: "train" || "eval"
        dataset_loader: train_loader || val_loader
        metrics: only available in eval phase

        Notes: loss of last iter is taken as loss of that epoch
        """
        num_class = self.__options.SOLVER.MODEL.ARGS.num_classes
        total_loss = 0

        # Epoch training
        for index, batch in tqdm(enumerate(dataset_loader), total=len(dataset_loader), colour="cyan",
                                 desc=phase.capitalize()):
            imgs, labels = batch[0].to(self.__device), batch[1]
            
            # reset gradients prior to forward pass
            self.__optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                # forward pass
                pred_labels = self.__model(imgs)
                
                if num_class == 1:
                    # Shape: N1 -> N
                    pred_labels = sigmoid(pred_labels).squeeze(dim=1)
                    labels = labels.type(torch.FloatTensor).to(self.__device)
                else:
                    # Shape: NC -> N
                    pred_labels = softmax(pred_labels, dim=1)
                    labels = labels.type(torch.LongTensor).to(self.__device)
                
                # Update loss
                mini_batch_loss = self.__loss(pred_labels, labels)
                
                # backprop + optimize only if in training phase
                if phase == 'train':
                    mini_batch_loss.backward()
                    self.__optimizer.step()
                    self.__lr_schedulers.step(epoch=epoch + index / len(dataset_loader))

                # Update metrics only if eval phase
                if metrics is not None:
                    metrics = [metric.update(pred_labels, labels) for metric in metrics]

                        
            # Accumulate minibatch into total loss
            total_loss += mini_batch_loss.item()

        if metrics is not None:
            metrics_name = self.__options.METRICS.NAME_LIST

            for i in range(len(metrics_name)):
                metrics[i] = metrics[i].compute()
                if isinstance(metrics[i], torch.Tensor):
                    metrics[i] = metrics[i].item() if metrics[i].dim() == 1 and len(metrics[i]) == 1 else metrics[i].detach().cpu().numpy().tolist()

                elif isinstance(metrics[i], tuple):
                    metrics[i] = [ele.detach().cpu().numpy().tolist() for ele in metrics[i]]

            training_result = {**{"loss": total_loss / len(dataset_loader)},
                               **{metric_name: value for metric_name, value in zip(metrics_name, metrics)}
                               }
        else:
            training_result = {"loss": total_loss / len(dataset_loader)}
        return training_result

    def __save_checkpoint(self, epoch: int, val_loss: float, obj: dict, save_all: bool = False) -> None:
        """
        save_all:
            True: save all trained epoch
            False: save only last and the best trained epoch
        Best_epoch is still saved in either save_all is True or False
        """
        save_name = os.path.join(self.__checkpoint_path, f"epoch_{epoch}.pt")
        torch.save(obj=obj, f=save_name)

        # Save best checkpoint
        if val_loss < self.__best_val_loss:
            save_name = os.path.join(self.__checkpoint_path, f"best_checkpoint.pt")
            torch.save(obj=obj, f=save_name)

            # Update best accuracy
            self.__best_val_loss = val_loss

        if not save_all and epoch - 1 > 0:
            # Remove previous epoch
            os.remove(os.path.join(self.__checkpoint_path, f"epoch_{epoch - 1}.pt"))
        return None

    def __get_best_val_loss(self) -> float:
        if "best_checkpoint.pt" in os.listdir(self.__checkpoint_path):
            return torch.load(f=os.path.join(self.__checkpoint_path, "best_checkpoint.pt"))["val_loss"]
        else:
            return float("inf")
