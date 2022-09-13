import os
import yaml
import time
import argparse
from easydict import EasyDict as edict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Running Experiments of Deep Prediction"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="config/resnet101_cifar.json",
        required=True,
        help="Path of config file",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="INFO",
        help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL",
    )
    parser.add_argument("-m", "--comment", help="Experiment comment")
    parser.add_argument("-t", "--test", help="Test model", action="store_true")
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--output-predictions", type=str, required=False, default=None)
    parser.add_argument(
        "--test-total-size",
        type=int,
        default=1000,
        help="Number of graphs to generate.",
    )
    parser.add_argument("--save-model", default=None, type=str)
    args = parser.parse_args()

    return args


def get_config(config_file, dataset_file, exp_dir=None, is_test=False):
    """ Construct and snapshot hyper parameters """
    # config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
    config = edict(yaml.load(open(config_file, "r")))

    # create hyper parameters
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    config.run_id = str(os.getpid())
    config.exp_name = "_".join(
        [
            config.model.name,
            dataset_name,
            time.strftime("%Y-%b-%d-%H-%M-%S"),
            config.run_id,
        ]
    )

    if exp_dir is not None:
        config.exp_dir = exp_dir

    if config.train.is_resume and not is_test:
        config.save_dir = config.train.resume_dir
        save_name = os.path.join(
            config.save_dir, "config_resume_{}.yaml".format(config.run_id)
        )
    else:
        config.save_dir = config.exp_dir
        save_name = os.path.join(config.save_dir, "config.yaml")

    # snapshot hyperparameters
    mkdir(config.save_dir)

    yaml.dump(edict2dict(config), open(save_name, "w"), default_flow_style=False)
    config["dataset"]["name"] = dataset_name

    return config


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
