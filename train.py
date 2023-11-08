import torch
from torch_geometric import seed
import torch.nn.functional as F
from torch_geometric.nn import WLConv
from model import GIN
from dataset_wl import dataset_gen, dataloader_gen, trial_loader
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

def model_training(hidden_size, num_layers, it, batch_size, threshold, trial_dataset = False):
    
    dataset = './data/QM9_IT'+str(num_layers)+'_THR'+str(threshold)+'/'  
    if trial_dataset:
        num_features = 1
        num_classes = 2
        train_loader = trial_loader(batch_size)
    else:
        train_loader, num_features, num_classes = dataloader_gen(dataset, num_layers, batch_size, threshold)     
    lr = 1e-3
    model = GIN(num_features, hidden_size, 1, 
                 num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 500
    raw_data = []

    @torch.enable_grad()
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def test(loader):
        model.eval()
        total_correct = 0
        tot_loss = 0

        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(pred, data.y)
            tot_loss += loss.item() * data.num_graphs
            total_correct += int((torch.abs(pred-data.y)<1/(2*(num_classes-1))).sum())
        return tot_loss / len(loader.dataset), total_correct/len(loader.dataset)*100
    max_correct = 0
    for epoch in range(1, epochs + 1):
        train()     
        train_loss, total_correct = test(train_loader)
        if total_correct > max_correct:
            max_correct = total_correct
        if epoch%500==0:
            print(f'epoch {epoch}')
            print(f'Train loss: {train_loss}, Train accuracy: {total_correct}%')
        raw_data.append({'Epoch': epoch, 'Train loss': train_loss, 'GNN hidden dimension': hidden_size, 
                         'train accuracy':total_correct, 'iteration':it})
        if total_correct == 100:
            print(f'epoch {epoch}')
            print(f'Train loss: {train_loss}, Train accuracy: {total_correct}%')
            return raw_data, total_correct
    
    return raw_data, max_correct


def main1():
    num_layers = 4
    batch_size = 64
    threshold = 35
    percentage = []
    hidden_size_list = [4,8,16,32,64]
    for hidden_size in hidden_size_list:
        raw = []
        percentage_list = []
        for it in range(15):
            seed.seed_everything(10*(it+1))
            raw_data, max_correct = model_training(hidden_size, num_layers, it, batch_size, threshold)
            raw+=raw_data
            percentage_list.append({'correct':max_correct, 'threshold': threshold, 'hidden_size':hidden_size, 'it':it})
        # data = pd.DataFrame.from_records(raw_data)
        # data.to_csv('static_GNN_IT'+str(num_layers)+'_THR'+str(threshold)+'.csv')
        percentage+=percentage_list
    train_acc_data = pd.DataFrame(percentage)
    train_acc_data.to_csv('train_acc_data_THR'+str(threshold)+'.csv')

def main2():
    hidden_size = 16
    num_layers = 4
    batch_size = 64
    threshold_list = [45-i for i in range(16)]
    percentage = []
    for threshold in threshold_list:
        raw = []
        percentage_list = []
        for it in range(15):
            seed.seed_everything(10*(it+1))
            raw_data, max_correct = model_training(hidden_size, num_layers, it, batch_size, threshold)
            raw+=raw_data
            percentage_list.append({'correct':max_correct, 'threshold': threshold, 'hidden_size':hidden_size, 'it':it})
        # data = pd.DataFrame.from_records(raw_data)
        # data.to_csv('static_GNN_IT'+str(num_layers)+'_THR'+str(threshold)+'.csv')
        percentage+=percentage_list
    train_acc_data = pd.DataFrame(percentage)
    train_acc_data.to_csv('train_acc_data_THR'+str(threshold)+'.csv')


if __name__ == '__main__':
    main1()
    main2()