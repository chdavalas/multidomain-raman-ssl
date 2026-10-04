#!/usr/bin/env python3

from lightning.pytorch.cli import LightningCLI
import logging

def main():
    level = logging.INFO
    logging.basicConfig(level=level)

    cli = LightningCLI(save_config_kwargs={"overwrite": False})

if __name__ == "__main__":
    main()
