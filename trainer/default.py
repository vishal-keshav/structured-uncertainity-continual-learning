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

def train_epoch(model, train_loader, optimizer):
    epoch_loss = 0.0
    criterion = nn.NLLLoss()
    model.train()
    nr_batches = 0
    for _, (x,y) in enumerate(train_loader):
        nr_batches += 1
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    return epoch_loss/float(nr_batches)

def accuracy_epoch(model, test_loader, optimizer):
    model.eval()
    correct_cnt, total_cnt = 0, 0
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1, keepdim=True)
            correct_cnt += pred.eq(y.view_as(pred)).sum().item()
            total_cnt += x.data.size()[0]
    return float(correct_cnt)/float(total_cnt)


def train(c, m, d, e):
    e.info("[INFO] Starting experiment " + e.get_exp_name() \
        + " on " + str(datetime.now()))
    e.info("[INFO] Device selected" + str(device))

    data_path = '../data'
    model_path = '../models'

    learning_rate = c['learning_rate']
    nr_epochs = c['nr_epochs']
    batch_size = c['batch_size']

    e.info("[INFO] " + stringify([str(k)+'_'+str(c[k]) for k in c]))

    train_loader, test_loader = d(data_path).get_data(batch_size)
    model = m()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(1, nr_epochs)):
        e.info("[INFO] Starting epoch " + str(epoch))
        train_loss = train_epoch(model, train_loader, optimizer)
        e.log("train_loss", train_loss)
        test_accuracy = accuracy_epoch(model, test_loader, optimizer)
        e.log("test_accuracy", test_accuracy)

    model_out_path = os.path.join(model_path, e.get_exp_name())
    if not os.path.exists(model_out_path): os.makedirs(model_out_path)
    torch.save(model.state_dict(), os.path.join(model_out_path, model.get_name()))
    e.info("[INFO] Model saved")

##################################### Test #####################################
def test():
    config = {'learning_rate': 0.001,
              'nr_epochs': 5,
              'batch_size': 128}
    from model.default import model as model_def
    from dataset.default import dataset
    from utils.log_utils import file_logger
    experiment = file_logger()
    train(config, model_def, dataset, experiment)

if __name__ == "__main__":
    test()