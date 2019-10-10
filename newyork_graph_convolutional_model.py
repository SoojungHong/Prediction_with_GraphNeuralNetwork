import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import time

from pylab import *
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from tqdm import tqdm


#-------------------------
#   data
#-------------------------

nyc_places_small = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_small.csv"
nyc_places_with_id = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_with_id.csv"
nyc_places_embedding_small = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_embedding_small.csv'
nyc_small_adj_matrix = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_small_adj_matrix.csv'
nyc_embedding_full = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_embedding_full.csv'
nyc_full_adj_matrix = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_full_adj_matrix.csv'



#-----------------------------------------------------
# one hot vector representation for level 1 category
#-----------------------------------------------------

max_value_index_to_vector = {0: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             1: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             2: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                             3: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
                             4: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                             5: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                             6: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                             7: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
                             8: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                             9: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
                             10: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}


def vector_to_label_val(vector_label):
    if (vector_label == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])).all():
        ret = 100
    elif (vector_label == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])).all():
        ret = 200
    elif (vector_label == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])).all():
        ret = 300
    elif (vector_label == np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])).all():
        ret = 350
    elif (vector_label == np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])).all():
        ret = 400
    elif (vector_label == np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])).all():
        ret = 500
    elif (vector_label == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])).all():
        ret = 550
    elif (vector_label == np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])).all():
        ret = 600
    elif (vector_label == np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])).all():
        ret = 700
    elif (vector_label == np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all():
        ret = 800
    else:
        ret = 900
    return ret


#-------------------------------
# helper functions for parsing
#-------------------------------
def embedding_string_to_vector(str):
    convt = str.replace('\n', '').replace('[ ', '').replace('[', '').replace(']', '').replace('  ', ' ').replace(' ', ',').split(',')
    ret = [float(i) for i in convt]

    return ret


def label_string_to_vector(str):
    convt = str.replace('\n', '').replace('[ ', '').replace('[', '').replace(']', '').replace('  ', ' ').replace(' ', ',').split(',')
    ret = [int(i) for i in convt]
    return ret


#------------------------------
# construct adjacency matrix
#------------------------------

def construct_nyc_dataframe():
    edges = pd.read_csv(nyc_full_adj_matrix)
    edges.columns = ['edge_id', 'start_pid', 'end_pid']

    raw_data = pd.read_csv(nyc_embedding_full)
    node_feature = raw_data['place_embedding']
    label_str_list = raw_data['label_encoding']

    start_node_list = edges['start_pid']
    end_node_list = edges['end_pid']

    raw_data['input_feature'] = node_feature.apply(lambda row: embedding_string_to_vector(row))
    raw_x = raw_data['input_feature']

    raw_data['label_vector'] = label_str_list.apply(lambda row: label_string_to_vector(row))
    raw_y = raw_data['label_vector']
    return start_node_list, end_node_list, raw_x, raw_y


def visualize_loss(epoch_index, loss_vals):
    plot(epoch_index, loss_vals)
    xlabel('epoch')
    ylabel('loss')
    title('Loss')
    grid(True)
    show()
    plt.savefig('foo.pdf')


def construct_pt():
    ny_start_nodes, ny_end_nodes, ny_raw_x, ny_raw_y = construct_nyc_dataframe()
    ny_edge_index = torch.tensor([ny_start_nodes, ny_end_nodes], dtype=torch.long)
    ny_x = torch.tensor(ny_raw_x, dtype=torch.float)
    ny_y = torch.tensor(ny_raw_y, dtype=torch.long)
    dataset = Data(x=ny_x, edge_index=ny_edge_index, y=ny_y)

    # save tensors
    torch.save(ny_edge_index, 'ny_full_edge_index_tensor.pt')
    torch.save(ny_x, 'ny_full_x_tensor.pt')
    torch.save(ny_y, 'ny_full_y_tensor.pt')


# load Chicago tensors - x, y and edges
edge_index = torch.load('ny_full_edge_index_tensor.pt')
x = torch.load('ny_full_x_tensor.pt')
y = torch.load('ny_full_y_tensor.pt')
dataset = Data(x=x, edge_index=edge_index, y=y)

nyc_places_num = dataset.num_nodes
idx = np.arange(nyc_places_num) #248023)
np.random.shuffle(idx)
num_test_data = int((nyc_places_num * 20)/100)
val_idx = idx[:num_test_data]
train_idx = idx[num_test_data:]

dataset.train_mask = torch.zeros(dataset.num_nodes, dtype=torch.uint8)
dataset.train_mask[train_idx] = 1  # train only on the 80% nodes
dataset.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.uint8)  # test on 20 % nodes
dataset.test_mask[val_idx] = 1


#---------------------
# Build Network
#---------------------
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)  # parameters - in_channel, out_channel, the other parameters are booleans
        self.conv2 = GCNConv(16, 11)  # ToDo : need to set num_classes (e.g. 11) - if I use InMemoryDataset, then num_classes would be captured
        self.linear1 = torch.nn.Linear(dataset.num_node_features, 16)
        self.linear2 = torch.nn.Linear(16, 11)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class Net7(torch.nn.Module):
    def __init__(self):
        super(Net7, self).__init__()

        self.conv1 = GCNConv(dataset.num_node_features, 256)
        self.conv2 = GCNConv(126, 62)
        self.linear1 = torch.nn.Linear(256, 11)
        self.linear2 = torch.nn.Linear(126, 11)
        self.pool1 = TopKPooling(512, ratio=0.5)
        self.pool2 = TopKPooling(62, ratio=0.5)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(62)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.linear1(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net7().to(device)
data = dataset.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

epoch_list = []
loss_list = []
model.train()
for epoch in range(2000):
    print('epoch : ', epoch)
    epoch_list.append(epoch)
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, torch.max(data.y, 1)[1])
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()


#----------------------------
# visualize loss per epoch
#----------------------------
visualize_loss(epoch_list, loss_list)


#-------------------
# model evaluation
#-------------------
model.eval()
_, pred = model(data).max(dim=1)

pred_test = pred[data.test_mask]
pred_test_encoded = []
correct = 0

for i in range(len(pred_test)):
    pred_vec = max_value_index_to_vector[pred[data.test_mask][i].item()]
    cuda0 = torch.device('cuda:0')
    cpu = torch.device('cpu')

    if torch.all(torch.eq(torch.tensor(pred_vec, dtype=torch.long, device=cuda0), data.y[data.test_mask][i])):
        correct = correct + 1
    else:
      print('nyc_poi index :', val_idx[i])

print('correct : ', correct)
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))

# save model
saved_model = 'saved_model.pkl'
torch.save(model, saved_model)
print('model saved')
