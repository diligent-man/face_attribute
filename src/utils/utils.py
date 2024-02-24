import os

from typing import Tuple, Dict, List
from src.modelling.vgg import get_vgg_model
from src.modelling.resnet import get_resnet_model

import torch
import torcheval
from torchsummary import summary


from torchvision.datasets import ImageFolder
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import DataLoader, random_split, Dataset
from torcheval.metrics import MulticlassF1Score, BinaryF1Score, MulticlassAccuracy, BinaryAccuracy
from torch.optim import Adam, AdamW, NAdam, RAdam, SparseAdam, Adadelta, Adagrad, Adamax, ASGD, RMSprop, Rprop, LBFGS, SGD
from torch.nn.modules import NLLLoss, NLLLoss2d, CTCLoss, KLDivLoss, GaussianNLLLoss, PoissonNLLLoss, L1Loss, MSELoss, HuberLoss, SmoothL1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ChainedScheduler, SequentialLR, ReduceLROnPlateau, OneCycleLR


__all__ = ["init_loss", "init_lr_scheduler", "init_metrics", "init_model_optimizer_start_epoch", "get_train_set", "get_test_set", "get_model_summary"]


def init_model(device: str, pretrained: bool, base: str,
               name: str, state_dict: dict, **kwargs) -> torch.nn.Module:
    available_bases = {
        "vgg": get_vgg_model,
        "resnet": get_resnet_model
    }
    assert base in available_bases.keys(), "Your selected base is unavailable"
    model = available_bases[base](device, name, pretrained, state_dict, **kwargs)
    return model


def init_optimizer(name: str, model_paras, state_dict: Dict = None, **kwargs) -> torch.optim.Optimizer:
    available_optimizers = {
        "Adam": Adam, "AdamW": AdamW, "NAdam": NAdam, "Adadelta": Adadelta, "Adagrad": Adagrad, "Adamax": Adamax,
        "RAdam": RAdam, "SparseAdam": SparseAdam, "RMSprop": RMSprop, "Rprop": Rprop, "ASGD": ASGD, "LBFGS": LBFGS,
        "SGD": SGD
    }
    assert name in available_optimizers.keys(), "Your selected optimizer is unavailable."

    # init optimizer
    optimizer: torch.optim.Optimizer = available_optimizers[name](model_paras, **kwargs)

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
    return optimizer


def init_model_optimizer_start_epoch(device: str,
                                     checkpoint_load: bool, checkpoint_path: str, resume_name: str,
                                     optimizer_name: str, optimizer_args: Dict,
                                     model_base: str, model_name: str, model_args: Dict,
                                     pretrained: bool = False
                                     ) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    model_state_dict = None
    optimizer_state_dict = None
    start_epoch = 1

    if checkpoint_load:
        checkpoint = torch.load(f=os.path.join(checkpoint_path, resume_name), map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model_state_dict = checkpoint["model_state_dict"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]

    model: torch.nn.Module = init_model(device=device, pretrained=pretrained, base=model_base,
                                        name=model_name, state_dict=model_state_dict, **model_args
                                        )

    optimizer: torch.optim.Optimizer = init_optimizer(name=optimizer_name, model_paras=model.parameters(),
                                                      state_dict=optimizer_state_dict, **optimizer_args
                                                      )
    return start_epoch, model, optimizer


def init_lr_scheduler(name: str, args: Dict, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    available_lr_scheduler = {
        "LambdaLR": LambdaLR, "MultiplicativeLR": MultiplicativeLR, "StepLR": StepLR, "MultiStepLR": MultiStepLR,
        "ConstantLR": ConstantLR,
        "LinearLR": LinearLR, "ExponentialLR": ExponentialLR, "PolynomialLR": PolynomialLR,
        "CosineAnnealingLR": CosineAnnealingLR,
        "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts, "ChainedScheduler": ChainedScheduler,
        "SequentialLR": SequentialLR,
        "ReduceLROnPlateau": ReduceLROnPlateau, "OneCycleLR": OneCycleLR
    }
    assert name in available_lr_scheduler.keys(), "Your selected lr scheduler is unavailable"
    return available_lr_scheduler[name](optimizer, **args)


def init_metrics(name_lst: List[str], args: Dict, device: str) -> List[torcheval.metrics.Metric]:
    available_metrics = {
        "BinaryAccuracy": BinaryAccuracy,
        "BinaryF1Score": BinaryF1Score,

        "MulticlassAccuracy": MulticlassAccuracy,
        "MulticlassF1Score": MulticlassF1Score
    }

    # check whether metrics available or not
    for metric in name_lst:
        assert metric in available_metrics.keys(), "Your selected metric is unavailable"

    metrics: List[torcheval.metrics.Metric] = []
    for i in range(len(name_lst)):
        metrics.append(available_metrics[name_lst[i]](**args[str(i)]))

    metrics = [metric.to(device) for metric in metrics]
    return metrics


def init_loss(name: str, args: Dict) -> torch.nn.Module:
    available_loss = {
        "NLLLoss": NLLLoss, "NLLLoss2d": NLLLoss2d,
        "CTCLoss": CTCLoss, "KLDivLoss": KLDivLoss,
        "GaussianNLLLoss": GaussianNLLLoss, "PoissonNLLLoss": PoissonNLLLoss,
        "CrossEntropyLoss": CrossEntropyLoss, "BCELoss": BCELoss, "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "L1Loss": L1Loss, "MSELoss": MSELoss, "HuberLoss": HuberLoss, "SmoothL1Loss": SmoothL1Loss,
    }
    assert name in available_loss.keys(), "Your selected loss function is unavailable"
    loss: torch.nn.Module = available_loss[name](**args)
    return loss


def get_train_set(root: str, input_size: int,
                  train_size: float, batch_size: int,
                  seed: int, cuda: bool, num_workers=1
                  ) -> Tuple[DataLoader, DataLoader]:
    # img -- reduce to 20% --> upsample to 224x224 -> Blur with ksize (10, 10)
    # Make female samples equal male

    # Use page-locked or not
    pin_memory = True if cuda is True else False

    dataset: Dataset = ImageFolder(root=os.path.join(root, "train"),
                                   transform=v2.Compose([
                                       # img from celeb A: 178 x 218 x 3
                                       v2.Resize(size=(int(178 * .2), int(218 * .2)), interpolation=InterpolationMode.NEAREST),
                                       v2.Resize(size=(input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                                       v2.GaussianBlur(kernel_size=5),
                                       v2.PILToTensor(),
                                       v2.ToDtype(torch.float32, scale=True)
                                   ]))

    train_set, validation_set = random_split(dataset=dataset,
                                             generator=torch.Generator().manual_seed(seed),
                                             lengths=[round(len(dataset) * train_size),
                                                      len(dataset) - round(len(dataset) * train_size)
                                                      ])

    train_set = DataLoader(dataset=train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory
                           )

    validation_set = DataLoader(dataset=validation_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory
                                )
    return train_set, validation_set


def get_test_set(root: str, input_size: int, batch_size: int, cuda: bool, num_workers=1) -> DataLoader:
    # Use page-locked or not
    pin_memory = True if cuda is True else False

    test_set: Dataset = ImageFolder(root=os.path.join(root, "test"),
                                    transform=v2.Compose([
                                        v2.Resize(size=(input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                                        v2.PILToTensor(),
                                        v2.ToDtype(torch.float32, scale=True)
                                    ]))

    test_set: DataLoader = DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory
                                      )
    return test_set


def get_model_summary(model: torch.nn.Module, input_size: Tuple):
    return summary(model, input_size)
