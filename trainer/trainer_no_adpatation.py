import os
import sys
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

sys.path.append(os.path.abspath('.'))
from utils.generic_utils import stringify

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

def train_epoch(model, x, y, opti, t, batch_size):
    epoch_loss = 0
    nr_examples = len(x)
    epoch_loss = 0.0
    criterion = nn.NLLLoss()
    model.train()
    for i in range(0, nr_examples, batch_size):
        x_data = x[i:i+batch_size]
        target = y[i:i+batch_size]
        opti.zero_grad()
        out = model(x_data)[t]
        loss = criterion(out, target)
        loss.backward()
        opti.step()
        epoch_loss+=loss.item()
    return epoch_loss/(batch_size+1)

def accuracy_epoch(model, x, y, t):
    model.eval()
    total_acc = 0
    total_num = 0
    nr_examples = len(x)
    batch_size = 128
    with torch.no_grad():
        num_batches = nr_examples//batch_size
        for i in range(0, nr_examples, batch_size):
            x_data = x[i:i+batch_size]
            target = y[i:i+batch_size]
            out = model(x_data)[t]
            _, pred = out.max(1, keepdim=True)
            total_acc += pred.eq(target.view_as(pred)).sum().item()
            total_num += len(x_data)
    return total_acc/total_num


def train(c, m, d, e):
    e.info("[INFO] Starting experiment " + e.get_exp_name() \
        + " on " + str(datetime.now()))
    e.info("[INFO] Device selected " + str(device))

    data_path = '../data'
    model_path = '../models'

    learning_rate = c['learning_rate']
    nr_epochs = c['nr_epochs']
    batch_size = c['batch_size']

    e.info("[INFO] " + stringify([str(k)+'_'+str(c[k]) for k in c]))
    dataset = d(data_path)
    model = m(28*28, 5, 2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for task_id in tqdm(range(5)):
        e.info("[INFO] Started training on task " + str(task_id))
        train_x, train_y = dataset.get_train_split(task_id)
        test_x, test_y = dataset.get_test_split(task_id)
        for epoch in tqdm(range(1, nr_epochs+1)):
            train_loss = train_epoch(model, train_x, train_y,
                optimizer, task_id, batch_size)
            e.log("train_loss_" + str(task_id), train_loss)
        for t in tqdm(range(task_id+1)):
            test_x, test_y = dataset.get_test_split(task_id)
            test_accuracy = accuracy_epoch(model, test_x, test_y, t)
            e.log("accuracy_"+str(task_id)+" for task "+ str(t), test_accuracy)

    model_out_path = os.path.join(model_path, e.get_exp_name())
    if not os.path.exists(model_out_path): os.makedirs(model_out_path)
    torch.save(model.state_dict(), os.path.join(model_out_path, model.get_name()))
    e.info("[INFO] Model saved")


##################################### Test #####################################
def test():
    config = {'learning_rate': 0.001,
              'nr_epochs': 20,
              'batch_size': 64}
    sys.path.append(os.path.abspath('.'))
    from model.multi_headed_cnn import model as model_def
    from dataset.split_5_mnist import dataset
    from utils.log_utils import file_logger
    experiment = file_logger()
    train(config, model_def, dataset, experiment)

if __name__ == "__main__":
    test()