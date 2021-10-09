from utils.common import read_config
import argparse
from utils.data_mgmt import get_data
from utils.model import create_model, save_model, save_plots
import os
import pandas as pd
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(logs_dir, 'ANN.log'), level=logging.INFO, format=logging_str,filemode='a')

def training(filename):
    config = read_config(filename)
    logging.info(config)
#    validation_datasize = config.get('validation_datasize')
    validation_datasize = config['params']['validation_datasize']
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

#    LOSS,OPTIMIZER,METRICS,OUTPUT_CLASSES = [config.get(key) for key in ['loss_function','optimizer','metrics','no_classes']]
    LOSS = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    OUTPUT_CLASSES = config['params']['no_classes']

    model = create_model(LOSS,OPTIMIZER,METRICS,OUTPUT_CLASSES)

    EPOCHS = config['params']['epochs']

    VALIDATION = (X_valid,y_valid)
    run_stats = model.fit(x=X_train, y=y_train , epochs=EPOCHS, validation_data=VALIDATION)
    df = pd.DataFrame(run_stats.history)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    plots_dir = config["artifacts"]["plots_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    plots_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plots_dir_path, exist_ok=True)

    model_name = config["artifacts"]["model_name"]
    plots_name = config["artifacts"]["plots_name"]
    save_model(model, model_name, model_dir_path)
    save_plots(df, plots_name, plots_dir_path)

    logging.info(model.evaluate(X_test,y_test))
    logging.info(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Artificial Neural Networks")
    parser.add_argument("--config", "-c",default="config.yaml")
    parsed_args = parser.parse_args()
    
    training(filename=parsed_args.config)
