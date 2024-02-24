import os
import commentjson

from box import Box
from src.tools.train import Trainer
from src.tools.inference import inference
from src.utils.utils import get_train_set, get_test_set

from torch.utils.data import DataLoader


def train(option_path: str) -> None:
    # Load dataset
    options = Box(commentjson.loads(open(file=option_path, mode="r").read()))

    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", options.SOLVER.MODEL.NAME)
    log_path = os.path.join(os.getcwd(), "logs", options.SOLVER.MODEL.NAME)

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path, 0x777, True)
        print(f"Checkpoint dir for {options.SOLVER.MODEL.NAME} was created.")

    if not os.path.isdir(log_path):
        os.makedirs(log_path, 0x777, True)
        print(f"Log dir checkpoint for {options.SOLVER.MODEL.NAME} was created.")

    train_log_path = os.path.join(log_path, f"training_log.json")
    eval_log_path = os.path.join(log_path, f"eval_log.json")

    train_loader, validation_loader = get_train_set(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME),
                                                    input_size=options.DATA.INPUT_SHAPE[0],
                                                    train_size=options.DATA.TRAIN_SIZE,
                                                    batch_size=options.DATA.BATCH_SIZE,
                                                    seed=options.MISC.SEED, cuda=options.MISC.CUDA,
                                                    num_workers=options.DATA.NUM_WORKERS)

    print(f"""Train batch: {len(train_loader)}, Validation batch: {len(validation_loader)}
Training model {options.SOLVER.MODEL.NAME}
""")

    trainer = Trainer(options=options,
                      train_log_path=train_log_path,
                      eval_log_path=eval_log_path,
                      checkpoint_path=checkpoint_path,
                      train_loader=train_loader,
                      validation_loader=validation_loader
                      )
    trainer.train(metric_in_train=True)
    return None


def test(option_path: str) -> None:
    options = Box(commentjson.loads(open(file=option_path, mode="r").read()))
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", options.MODEL.NAME, options.CHECKPOINT.NAME)
    log_path = os.path.join(os.getcwd(), "logs", options.MODEL.NAME, "testing_log.json")

    test_loader: DataLoader = get_test_set(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME),
                                           input_size=options.DATA.INPUT_SHAPE[0],
                                           batch_size=options.DATA.BATCH_SIZE,
                                           cuda=options.MISC.CUDA,
                                           num_workers=options.DATA.NUM_WORKERS
                                           )
    print(f"""Test batch: {len(test_loader)}""")

    options: Box = Box(commentjson.loads(open(file=option_path, mode="r").read()))
    inference(options=options, checkpoint_path=checkpoint_path, log_path=log_path, test_loader=test_loader)
    return None


def main() -> None:
    # train(option_path=os.path.join(os.getcwd(), "configs", "vgg_train_config.json"))
    test(option_path=os.path.join(os.getcwd(), "configs", "inference_config.json"))


    # training_log_visualization(file_name="vgg13_training_log.json",
    #                        metrics_lst=["loss", "acc", "f1"],
    #                        base_name="vgg13"
    #                        )
    #
    # To-do list
    # Model evaluator
    # Visualization
    return None


if __name__ == '__main__':
    main()
