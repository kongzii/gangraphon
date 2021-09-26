import os
import sys
import torch
import logging
import traceback
import numpy as np
from pprint import pprint

from .runner import *
from pathlib import Path
from .utils.logger import setup_logging
from .utils.arg_helper import parse_arguments, get_config

torch.set_printoptions(profile="full")


def main():
    args = parse_arguments()
    config = get_config(
        args.config_file,
        is_test=args.test,
        dataset_file=args.dataset_file,
        exp_dir=args.save_model,
    )
    config["dataset"]["dataset_file"] = args.dataset_file
    if args.output_predictions is None:
        args.output_predictions = "/".join(args.dataset_file.split("/")[:-1]) + "/gran"
        Path(args.output_predictions).mkdir(exist_ok=True, parents=True)
    config["test"]["graph_save_path"] = args.output_predictions
    config["test"]["num_test_gen"] = args.test_total_size
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    config.use_gpu = config.use_gpu and torch.cuda.is_available()

    # log info
    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print(">" * 80)
    pprint(config)
    print("<" * 80)

    # Run the experiment
    try:
        runner = eval(config.runner)(config)
        if not args.test:
            runner.train()
        else:
            runner.test()
    except:
        logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
