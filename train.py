import argparse
import json

import lightning as L
import torchsummary

from dataloaders.myDatamodule import myDataModule
from models.myModel import myModel

L.seed_everything(911)


# Parsing command line arguments for configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Train a -----')
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the config file'
    )
    return parser.parse_args()


# Loading configuration from a JSON file
def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)


# Initializing the Lightning trainer with configurations
def initialize_trainer(config):
    return L.Trainer(
        max_epochs=config['max_epochs'],
        logger=L.pytorch.loggers.TensorBoardLogger(
            config['log_dir'], name=config['logger_name']
        ),
        fast_dev_run=config['fast_dev_run'],
    )


# Printing summary of the model
def print_model_summary(model, config):
    print('Generator Summary:')
    torchsummary.summary(model, (1, 1, 1), device='cpu')


def main(config_path):
    config = load_config(config_path)

    dm = myDataModule(config['datamodule'])

    model = myModel(config['model'])
    print_model_summary(model, config['model'])

    trainer = initialize_trainer(config['trainer'])

    trainer.fit(model, dm)


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
