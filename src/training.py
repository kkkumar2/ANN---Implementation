from utils.common import read_config
import argparse
from utils.data_mgmt import get_data
from utils.model import create_model, save_model, save_plots, save_tf_logs
import os
import pandas as pd
import logging
import time
import numpy as np
import tensorflow as tf




def training(filename):
    config = read_config(filename)
    logging.info(config)
#    validation_datasize = config.get('validation_datasize')

    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    logs_dir = config['logs']['logs_dir']
    general_logs = config['logs']['general_logs']
    os.makedirs(logs_dir, exist_ok=True)
    general_logs_dir = os.path.join(logs_dir, general_logs)
    os.makedirs(general_logs_dir, exist_ok=True)
    logging.basicConfig(filename = os.path.join(general_logs_dir, 'ANN.log'), level=logging.INFO, format=logging_str,filemode='a')

##
    validation_datasize = config['params']['validation_datasize']
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

#    LOSS,OPTIMIZER,METRICS,OUTPUT_CLASSES = [config.get(key) for key in ['loss_function','optimizer','metrics','no_classes']]
    LOSS = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    OUTPUT_CLASSES = config['params']['no_classes']

    model = create_model(LOSS,OPTIMIZER,METRICS,OUTPUT_CLASSES)

    EPOCHS = config['params']['epochs']
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    plots_dir = config["artifacts"]["plots_dir"]
    tensorboard_logs = config["logs"]["tensorboard_logs"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_chk_dir_path = os.path.join(artifacts_dir, model_dir, 'model_chk')
    os.makedirs(model_dir_path, exist_ok=True)

    plots_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plots_dir_path, exist_ok=True)

    tf_dir_path = os.path.join(logs_dir, tensorboard_logs)
    os.makedirs(tf_dir_path, exist_ok=True)

    model_name = config["artifacts"]["model_name"]
    plots_name = config["artifacts"]["plots_name"]
    tf_logs_name = config["logs"]["tf_logs_name"]

    tme = "log_%Y_%m_%d_%H_%M_%S"
    tf_writer, tf_logs = save_tf_logs(tf_dir_path, tf_logs_name, tme)
    with tf_writer.as_default():
        images = np.reshape(X_train[:20],(-1,28,28,1))
        tf.summary.image("Sample images from training dataset", images, max_outputs=20, step=0)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tf_logs)
    early_Stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    chk_path = os.path.join(model_chk_dir_path + '/model.h5')
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(chk_path, save_best_only=True)

    CALLBACKS_LIST = [tensorboard_cb, early_Stopping_cb, checkpointing_cb]

    VALIDATION = (X_valid,y_valid)
    run_stats = model.fit(x=X_train, y=y_train , epochs=EPOCHS, validation_data=VALIDATION,callbacks=CALLBACKS_LIST)
    df = pd.DataFrame(run_stats.history)

    tme = '%Y%m%d_%H%M%S_'
    save_model(model, model_name, model_dir_path,tme)
    save_plots(df, plots_name, plots_dir_path,tme)

    logging.info(model.evaluate(X_test,y_test))
    logging.info(f" \n, {df}")

    ckpt_model = tf.keras.models.load_model(chk_path)

    history = ckpt_model.fit(X_train, y_train, epochs=EPOCHS,validation_data=VALIDATION, callbacks=CALLBACKS_LIST)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Artificial Neural Networks")
    parser.add_argument("--config", "-c",default="config.yaml")
    parsed_args = parser.parse_args()
    
    training(filename=parsed_args.config)
