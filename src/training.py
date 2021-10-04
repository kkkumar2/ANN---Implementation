from src.utils.common import read_config
import argparse

def training(filename):
    config = read_config(filename)
    print(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Artificial Neural Networks")
    parser.add_argument("--config","-c",default="config.yaml")
    parsed_args = parser.parse_args()
    
    training(filename=parsed_args.config)
