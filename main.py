import random
import numpy as np
import pandas as pd
from net import Net
from dataset import make_data
from classifier import train, test, KL, nonRCT_KL
import torch
import torch.nn as nn
import torch.optim as optim
import shap
from datalist import make


SEED = 42
DATA_NAME = "normal"
DATA_SIZE = 100000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCH = 100


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def rct():
    data = make(DATA_NAME, DATA_SIZE)
    data.true_KL()
    train_loader = make_data(data, BATCH_SIZE)

    criterion = nn.BCEWithLogitsLoss()
    net = Net(data.in_dim+data.out_dim, 32, 0)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    train(train_loader, net, criterion, optimizer, EPOCH)
    test(train_loader, net, criterion)
    predict = KL(train_loader, net)
    print(f"predicted KL:{predict}")
    
    train_sample, _ = next(iter(train_loader))
    test_sample, _ = next(iter(train_loader))
    explainer = shap.DeepExplainer(net, train_sample)
    shap_values = explainer.shap_values(test_sample)
    test_sample = test_sample.detach().numpy()
    shap.summary_plot(shap_values, test_sample, feature_names=data.columns, plot_type="bar")
    test_sample = pd.DataFrame(test_sample, columns=data.columns)
    shap.dependence_plot("Out 0", shap_values, test_sample)
    shap.dependence_plot("Out 1", shap_values, test_sample)
    shap.dependence_plot("Out 2", shap_values, test_sample)
    shap.dependence_plot("Out 3", shap_values, test_sample)
    shap.dependence_plot("Out 4", shap_values, test_sample)

def non_rct():
    data = make(DATA_NAME, DATA_SIZE)
    data.true_KL()
    pros_loader, train_loader = make_data(data, BATCH_SIZE)

    criterion = nn.BCEWithLogitsLoss()
    pros_net = Net(data.in_dim, 32, 0)
    net = Net(data.in_dim + data.out_dim, 32, 0)
    pros_optimizer = optim.Adam(pros_net.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    train(pros_loader, pros_net, criterion, pros_optimizer, EPOCH)
    train(train_loader, net, criterion, optimizer, EPOCH)
    test(pros_loader, pros_net, criterion)
    test(train_loader, net, criterion)
    predict = nonRCT_KL(pros_loader, train_loader, data.in_dim, pros_net, net)
    print(f"predicted KL:{predict}")
    
    train_sample, _ = next(iter(train_loader))
    test_sample, _ = next(iter(train_loader))
    explainer = shap.DeepExplainer(net, train_sample)
    shap_values = explainer.shap_values(test_sample)
    test_sample = test_sample.detach().numpy()
    shap.summary_plot(shap_values, test_sample, feature_names=data.columns, plot_type="bar")
    test_sample = pd.DataFrame(test_sample, columns=data.columns)
    shap.dependence_plot("Out 0", shap_values, test_sample)
    shap.dependence_plot("Out 1", shap_values, test_sample)
    shap.dependence_plot("Out 2", shap_values, test_sample)
    shap.dependence_plot("Out 3", shap_values, test_sample)
    shap.dependence_plot("Out 4", shap_values, test_sample)

fix_seed(SEED)
rct()
# non_rct()