import os

import torch
import torcheval
from typing import List, Dict
from torchvision.transforms import v2


class TrainingArguments:
    # DATA in config file
    dataset_path: str = None
    input_shape: List[int] = [224, 224, 3]
    train_size: int = 0.9
    batch_size: int = 16
    num_worker: int = 1
    transform: v2.Compose = None
    target_transform: v2.Compose = None

    # CHECKPOINT in config file
    checkpoint_path: str = None
    save_strategy: str = "no"
    save_total_lim: int = 2
    load: bool = False
    resume_name: str = None
    save_only_weight: bool = True
    include_config: bool = True

    metrics: List[torcheval.metrics.Metric] = None
    model: torch.nn.Module = None

    # SOLVER in config file
    optimizer: torch.optim.Optimizer = torch.optim.AdamW()
    lr_scheduler: torch.optim.lr_scheduler = None
    loss: torch.nn.Module = None

    # EARLY_STOPPING in config file
    early_stopping_apply: bool = False

    # LOGGING in config file
    log_path: str = None
    logging_strategy: str = "epoch"

    # MISC
    seed: int = 12345
    device: str = "cpu"
    training_epochs: int = 3
    evaluation_strategy: str = "no"
    prediction_loss_only: bool = True
    """
    Parameters:
        evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are:
                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.
        prediction_loss_only (`bool`, *optional*, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        batch_size (int)
        training_epochs('int', defaults to 3):
        
        logging_dir (`str`, *optional*):
            Will default to:
                *output/log/backbone/derivative.
        logging_strategy ('str', defaults to "epoch"):
            - `"no"`: No logging is done during training.
            - `"epoch"`: Logging is done at the end of each epoch.
        
        save_strategy ('str', defaults to "epoch"):
            - `"no"`: No save is done during training.
            - `"epoch"`: Save is done at the end of each epoch.
        save_total_limit (`int`, *optional*):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`. When `load_best_model_at_end` is enabled, the "best" checkpoint according to
            `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for
            `save_total_limit=5` and `load_best_model_at_end`, the four last checkpoints will always be retained
            alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two
            checkpoints are saved: the last one and the best one (if they are different).
        save_only_weight ('bool', defaults to `False`):
            When checkpointing, whether to only save the model's weight, or entire model (weight + arch)
        cuda ('bool, defaults to `True`):
            Whether or not to use gpu. If set to False, we will use cpu or mps device if available.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
            
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        
        disable_tqdm (`bool`, *optional*):
            Whether or not to disable the tqdm progress bars and table of metrics produced by
            [`~notebook.NotebookTrainingTracker`] in Jupyter Notebooks. Will default to `True` if the logging level is
            set to warn or lower (default), `False` otherwise.
        
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Whether or not to load the best model found during training at the end of training. When this option is
            enabled, the best checkpoint will always be saved. See
            [`save_total_limit`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_total_limit)
            for more.

            <Tip>

            When set to `True`, the parameters `save_strategy` needs to be the same as `evaluation_strategy`, and in
            the case it is "steps", `save_steps` must be a round multiple of `eval_steps`.

            </Tip>

        metric_for_best_model (`str`, *optional*):
            Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`. Will
            default to `"loss"` if unspecified and `load_best_model_at_end=True` (to use the evaluation loss).

            If you set this value, `greater_is_better` will default to `True`. Don't forget to set it to `False` if
            your metric is better when lower.


        optimizer (dict[str, dict], defaults to "AdamW"):
        lr_scheduler (dict[str, dict], defaults to "CosineAnnealingWarmRestarts")
        resume_from_checkpoint (`str`, *optional*):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.

    ###########################################################################
    Implement later
        torch_compile (`bool`, *optional*, defaults to `False`):
            Whether or not to compile the model using PyTorch 2.0
            [`torch.compile`](https://pytorch.org/get-started/pytorch-2.0/).

            This will use the best defaults for the [`torch.compile`
            API](https://pytorch.org/docs/stable/generated/torch.compile.html?highlight=torch+compile#torch.compile).
            You can customize the defaults with the argument `torch_compile_backend` and `torch_compile_mode` but we
            don't guarantee any of them will work as the support is progressively rolled in in PyTorch.

            This flag and the whole compile API is experimental and subject to change in future releases.
        torch_compile_backend (`str`, *optional*):
            The backend to use in `torch.compile`. If set to any value, `torch_compile` will be set to `True`.

            Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

            This flag is experimental and subject to change in future releases.
        torch_compile_mode (`str`, *optional*):
            The mode to use in `torch.compile`. If set to any value, `torch_compile` will be set to `True`.

            Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

            This flag is experimental and subject to change in future releases.
    """
    def __init__(self,
                 dataset_path: str,
                 input_shape: List[int],
                 train_size: int,
                 batch_size: int,
                 num_worker: int,
                 transform: v2.Compose,
                 target_transform: v2.Compose,

                 checkpoint_path: str,
                 save_strategy: str,
                 save_total_lim: int,
                 load: bool,
                 resume_name: str,
                 save_only_weight: bool,
                 include_config: bool,

                 metrics: List[torcheval.metrics.Metric],
                 model: torch.nn.Module,

                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler,
                 loss: torch.nn.Module,

                 early_stopping_apply: bool,

                 log_path: str,
                 logging_strategy: str,

                 seed: int,
                 device: str,
                 training_epochs: int,
                 evaluation_strategy: str,
                 prediction_loss_only: bool
                 ):
        self.__dataset_path: str = dataset_path
        self.__input_shape: List[int] = input_shape
        self.__train_size: int = train_size
        self.__batch_size: int = batch_size
        self.__num_worker: int = num_worker
        self.__transform: v2.Compose = transform
        self.__target_transform: v2.Compose = target_transform

        self.__checkpoint_path: str = checkpoint_path
        self.__save_strategy: str = save_strategy
        self.__save_total_lim: int = save_total_lim
        self.__load: bool = load
        self.__resume_name: str = resume_name
        self.__save_only_weight: bool = save_only_weight
        self.__include_config: bool = include_config

        self.__metrics: List[torcheval.metrics.Metric] = metrics
        self.__model: torch.nn.Module = model

        self.__optimizer: torch.optim.Optimizer = optimizer
        self.__lr_scheduler: torch.optim.lr_scheduler = lr_scheduler
        self.__loss: torch.nn.Module = loss

        self.__early_stopping_apply: bool = early_stopping_apply

        self.__log_path: str = log_path
        self.__logging_strategy: str = logging_strategy

        self.__seed: int = seed
        self.__device: str = device
        self.__training_epochs: int = training_epochs
        self.__evaluation_strategy: str = evaluation_strategy
        self.__prediction_loss_only: bool = prediction_loss_only





























