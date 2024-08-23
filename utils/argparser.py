import argparse

# Function for parsing arguments
def arg_parse():

    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('-m',   '--model',             type=str,   default = 'LSTM',         help="Which model to use")
    parser.add_argument('-mn',  '--model_name',        type=str,   default = '',             help="Name of the model when saving it")
    parser.add_argument('-l',   '--load',              type=str,   default = '',             help="Path to the model you want to load")

    # LSTM and GRU model arguments
    parser.add_argument('-hs',  '--hidden_size',       type=int,   default = 64,             help="Size of the models hidden layer")
    parser.add_argument('-nl',  '--number_of_layers',  type=int,   default = 2,              help="Number of layers in the model")

    # Transformer model argumentds
    parser.add_argument('-md',  '--model_dimension',   type=int,   default = 64,             help="Model dimensions")
    parser.add_argument('-ah',  '--attention_heads',   type=int,   default = 4,              help="Number of attention heads in the model")

    # Training arguments
    parser.add_argument('-t',   '--train',             action="store_true", default = False, help="Weather to train the model")
    parser.add_argument('-e',   '--epochs',            type=int,   default = 5,              help="Number of epochs in training")
    parser.add_argument('-lr',  '--learning_rate',     type=float, default = 0.001,          help="Learning rate in training")

    # Data arguments
    parser.add_argument('-sl',  '--sequence_length',   type=int,   default = 100,            help="Length of the sequences that will be extracted")
    parser.add_argument('-ps',  '--prediction_step',   type=int,   default = 10,             help="Prediction step 1 = 10min, 100 = 100min")

    # Prediction argument
    parser.add_argument('-pt',  '--prediction_target', type=str,   default = 'PowerConsumption_Zone1', help="Which feature the model will be predicting")

    # Parse the arguments
    return parser.parse_args()
