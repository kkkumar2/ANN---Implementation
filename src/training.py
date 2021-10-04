from utils.common import read_config
import argparse
from utils.data_mgmt import get_data
from utils.model import create_model

def training(filename):
    config = read_config(filename)
    print(config)
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
    print(EPOCHS)
    print(validation_datasize)
    print(LOSS,OPTIMIZER,METRICS,OUTPUT_CLASSES)

    VALIDATION = (X_valid,y_valid)
    run_stats = model.fit(x=X_train, y=y_train , epochs=EPOCHS, validation_data=VALIDATION)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Artificial Neural Networks")
    parser.add_argument("--config", "-c",default="config.yaml")
    parsed_args = parser.parse_args()
    
    training(filename=parsed_args.config)
