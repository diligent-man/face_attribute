import os
import commentjson

from box import Box
from pprint import pprint as pp
from src.tools.train import Trainer
from src.tools.evaluate import evaluate
from src.utils.ConfigManager import ConfigManager
from src.utils.utils import get_train_val_loader, get_test_loader, get_dataset

from torch.utils.data import DataLoader


def train(configManager: ConfigManager) -> None:
    train_log_path = os.path.join(log_path, f"training_log.json")
    eval_log_path = os.path.join(log_path, f"eval_log.json")

    train_set = get_dataset(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME, "train"),
                            transform=options.DATA.TRANSFORM,
                            )

    train_loader, val_loader = get_train_val_loader(dataset=train_set,
                                                    train_size=options.DATA.TRAIN_SIZE,
                                                    batch_size=options.DATA.BATCH_SIZE, seed=options.MISC.SEED,
                                                    cuda=options.MISC.CUDA, num_workers=options.DATA.NUM_WORKERS
                                                    )
    print(f"""Train batch: {len(train_loader)}, Validation batch: {len(val_loader)}
Training model {options.SOLVER.MODEL.NAME}
""")

    trainer = Trainer(options=options,
                      train_log_path=train_log_path,
                      eval_log_path=eval_log_path,
                      checkpoint_path=checkpoint_path,
                      train_loader=train_loader,
                      val_loader=val_loader
                      )
    trainer.train(metric_in_train=True)
    return None


def test(option_path: str) -> None:
    for dataset in (["celeb_A", "collected_v3", "collected_v4"]):
        options = Box(commentjson.loads(open(file=option_path, mode="r").read()))
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints", options.MODEL.NAME, options.CHECKPOINT.NAME)

        options.DATA.DATASET_NAME = dataset
        log_path = os.path.join(os.getcwd(), "logs", options.MODEL.NAME, f"testing_log_{dataset}.json")

        test_set = get_dataset(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME, "test"),
                               transform=options.DATA.TRANSFORM,
                               )

        test_loader: DataLoader = get_test_loader(dataset=test_set,
                                                  batch_size=options.DATA.BATCH_SIZE,
                                                  cuda=options.MISC.CUDA,
                                                  num_workers=options.DATA.NUM_WORKERS
                                                  )
        print(f"""Test batch: {len(test_loader)}""")

        evaluate(options=options, checkpoint_path=checkpoint_path, log_path=log_path, test_loader=test_loader)
    return None


def main() -> None:
    # generate_celeb_A_dataset()
    config_path = os.path.join(os.getcwd(), "configs", "vgg.json")
    configManager = ConfigManager(config_path)
    train(configManager)
    # train(option_path=os.path.join(os.getcwd(), "configs", "age_config.json"))
    # test(option_path=os.path  .join(os.getcwd(), "configs", "test_config.json"))


    return None


if __name__ == '__main__':
    main()
