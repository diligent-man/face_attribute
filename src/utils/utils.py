import os

from box import Box
from typing import Tuple, Dict, List
from src.modelling.vgg import get_vgg_model
from src.modelling.resnet import get_resnet_model

import torch
import torcheval

from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode, Compose
from torch.utils.data import DataLoader, random_split, Dataset

from torch.optim import (
    Adam,
    AdamW,
    NAdam,
    RAdam,
    SparseAdam,
    Adadelta,
    Adagrad,
    Adamax,
    ASGD,
    RMSprop,
    Rprop,
    LBFGS,
    SGD
)

from torch.nn.modules import (
    NLLLoss,
    NLLLoss2d,
    CTCLoss,
    KLDivLoss,
    GaussianNLLLoss,
    PoissonNLLLoss,
    L1Loss,
    MSELoss,
    HuberLoss,
    SmoothL1Loss,
    CrossEntropyLoss,
    BCELoss,
    BCEWithLogitsLoss
)

from torcheval.metrics import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
    BinaryBinnedPrecisionRecallCurve,
    MulticlassBinnedPrecisionRecallCurve,
    BinaryPrecisionRecallCurve
)

from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    PolynomialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ChainedScheduler,
    SequentialLR,
    ReduceLROnPlateau,
    OneCycleLR
)

from torchvision.transforms.v2 import (
    # Color
    ColorJitter,
    Grayscale,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomChannelPermutation,
    RandomEqualize,
    RandomGrayscale,
    RandomInvert,
    RandomPhotometricDistort,
    RandomPosterize,
    RandomSolarize,

    # Geometry
    CenterCrop,
    ElasticTransform,
    FiveCrop,
    Pad,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomPerspective,
    RandomResize,
    RandomResizedCrop,
    RandomRotation,
    RandomShortestSize,
    RandomVerticalFlip,
    RandomZoomOut,
    Resize,
    ScaleJitter,
    TenCrop,

    # Meta
    ClampBoundingBoxes,
    ConvertBoundingBoxFormat,

    # Misc
    ConvertImageDtype,
    GaussianBlur,
    Identity,
    Lambda,
    LinearTransformation,
    Normalize,
    SanitizeBoundingBoxes,
    ToDtype,

    # Temporal
    UniformTemporalSubsample,

    # Type conversion
    PILToTensor,
    ToImage,
    ToPILImage,
    ToPureTensor,
    ToTensor
)


__all__ = ["get_dataset", "get_train_val_loader", "get_test_loader", "get_model_summary",
           "init_loss", "init_lr_scheduler", "init_metrics", "init_model", "init_model_optimizer_start_epoch"
           ]


def get_model_summary(model: torch.nn.Module, input_size: Tuple, device: str):
    return summary(model=model, input_size=input_size, device=device)


def get_transformation(transforms: Dict[str, Dict] = None) -> Compose:
    available_transforms = {
        # Color
        "ColorJitter": ColorJitter,
        "Grayscale": Grayscale,
        "RandomAdjustSharpness": RandomAdjustSharpness,
        "RandomAutocontrast": RandomAutocontrast,
        "RandomChannelPermutation": RandomChannelPermutation,
        "RandomEqualize": RandomEqualize,
        "RandomGrayscale": RandomGrayscale,
        "RandomInvert": RandomInvert,
        "RandomPhotometricDistort": RandomPhotometricDistort,
        "RandomPosterize": RandomPosterize,
        "RandomSolarize": RandomSolarize,

        # Geometry
        "CenterCrop": CenterCrop,
        "ElasticTransform": ElasticTransform,
        "FiveCrop": FiveCrop,
        "Pad": Pad,
        "RandomAffine": RandomAffine,
        "RandomCrop": RandomCrop,
        "RandomHorizontalFlip": RandomHorizontalFlip,
        "RandomIoUCrop": RandomIoUCrop,
        "RandomPerspective": RandomPerspective,
        "RandomResize": RandomResize,
        "RandomResizedCrop": RandomResizedCrop,
        "RandomRotation": RandomRotation,
        "RandomShortestSize": RandomShortestSize,
        "RandomVerticalFlip": RandomVerticalFlip,
        "RandomZoomOut": RandomZoomOut,
        "Resize": Resize,
        "ScaleJitter": ScaleJitter,
        "TenCrop": TenCrop,

        # Meta
        "ClampBoundingBoxes": ClampBoundingBoxes,
        "ConvertBoundingBoxFormat": ConvertBoundingBoxFormat,

        # Misc
        "ConvertImageDtype": ConvertImageDtype,
        "GaussianBlur": GaussianBlur,
        "Identity": Identity,
        "Lambda": Lambda,
        "LinearTransformation": LinearTransformation,
        "Normalize": Normalize,
        "SanitizeBoundingBoxes": SanitizeBoundingBoxes,
        "ToDtype": ToDtype,

        # Temporal
        "UniformTemporalSubsample": UniformTemporalSubsample,

        # Type conversion
        "PILToTensor": PILToTensor,
        "ToImage": ToImage,
        "ToPILImage": ToPILImage,
        "ToPureTensor": ToPureTensor,
        "ToTensor": ToTensor
    }

    # Took from InterpolationMode of pytorch
    available_interpolation = {
        "NEAREST": InterpolationMode.NEAREST,
        "NEAREST_EXACT": InterpolationMode.NEAREST_EXACT,
        "BILINEAR": InterpolationMode.BILINEAR,
        "BICUBIC": InterpolationMode.BICUBIC,
        # For PIL compatibility
        "BOX": InterpolationMode.BOX,
        "HAMMING": InterpolationMode.HAMMING,
        "LANCZOS": InterpolationMode.LANCZOS,
    }

    available_dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64
    }

    if transforms is not None:
        # Verify transformation
        for k in transforms.keys():
            assert k in available_transforms.keys(), "Your selected transform method is unavailable"

            # Verify interpolation mode & replace str name to its corresponding func
            if k in ("Resize", "RandomRotation"):
                assert transforms[k]["interpolation"] in available_interpolation.keys(), "Your selected interpolation mode in unavailable"
                transforms[k]["interpolation"] = available_interpolation[transforms[k]["interpolation"]]

            # Verify dtype & replace str name to its corresponding func
            if k in ("ToDtype"):
                assert transforms[k]["dtype"] in available_dtype.keys(), "Your selected dtype in unavailable"
                transforms[k]["dtype"] = available_dtype[transforms[k]["dtype"]]

        compose: Compose = Compose([available_transforms[k](**v) for k, v in transforms.items()])
    else:
        compose: Compose = Compose([])
    return compose


def get_dataset(root: str,
                transform: Dict[str, Dict] = None,
                target_transform: Dict[str, Dict] = None
                ) -> Dataset:
    """
    Args:
        root: dataset dir
        transform: Dict of transformation name and its corresponding kwargs
        target_transform:                     //                            but for labels
    """
    return ImageFolder(root=root,
                       transform=get_transformation(transforms=transform),
                       target_transform=get_transformation(transforms=target_transform)
                       )


def get_train_val_loader(dataset: Dataset,
                         train_size: float,
                         batch_size: int,
                         seed: int,
                         cuda: bool,
                         num_workers=1,
                         customDataloader=None
                         ) -> Tuple[DataLoader, DataLoader]:
    train_size = round(len(dataset) * train_size)
    pin_memory = True if cuda is True else False  # Use page-locked or not

    train_set, validation_set = random_split(dataset=dataset,
                                             generator=torch.Generator().manual_seed(seed),
                                             lengths=[train_size, len(dataset) - train_size])
    if customDataloader is None:
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
    else:
        train_set = customDataloader(dataset=train_set,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory
                                     )

        validation_set = customDataloader(dataset=validation_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory
                                          )
    return train_set, validation_set


def get_test_loader(dataset: Dataset, batch_size: int, cuda: bool, num_workers=1) -> DataLoader:
    # Use page-locked or not
    pin_memory = True if cuda is True else False
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=pin_memory
                      )
##########################################################################################################################


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
        "BinaryPrecision": BinaryPrecision,
        "BinaryRecall": BinaryRecall,
        "BinaryConfusionMatrix": BinaryConfusionMatrix,
        "BinaryPrecisionRecallCurve": BinaryPrecisionRecallCurve,
        "BinaryBinnedPrecisionRecallCurve": BinaryBinnedPrecisionRecallCurve,

        "MulticlassAccuracy": MulticlassAccuracy,
        "MulticlassF1Score": MulticlassF1Score,
        "MulticlassPrecision": MulticlassPrecision,
        "MulticlassRecall": MulticlassRecall,
        "MulticlassConfusionMatrix": MulticlassConfusionMatrix,
        "MulticlassBinnedPrecisionRecallCurve": MulticlassBinnedPrecisionRecallCurve
    }

    # check whether metrics available or not
    for metric in name_lst:
        assert metric in available_metrics.keys(), "Your selected metric is unavailable"

    metrics: List[torcheval.metrics.Metric] = []
    for i in range(len(name_lst)):
        metrics.append(available_metrics[name_lst[i]](**args[str(i)]))

    metrics = [metric.to(device) for metric in metrics]
    return metrics


def init_model(device: str, pretrained: bool, base: str,
               name: str, state_dict: dict, **kwargs) -> torch.nn.Module:
    available_bases = {
        "vgg": get_vgg_model,
        "resnet": get_resnet_model
    }
    assert base in available_bases.keys(), "Your selected base is unavailable"
    model: torch.nn.Module = available_bases[base](device, name, pretrained, state_dict, **kwargs)
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
