import os

import torch

from train import \
    DATA_DIR, \
    MODEL_FILE, \
    build_network, \
    train

if __name__ == '__main__':
    # create cuda device
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create the network
    model = build_network()

    # if true, try to restore the network from the data file
    restore = False
    if restore:
        model_path = os.path.join(DATA_DIR, MODEL_FILE)
        model.load_state_dict(torch.load(model_path))

    # transfer to the gpu
    model = model.to(dev)

    # train
    train(model, dev)