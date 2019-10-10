import numpy as np
import pandas as pd
import torch
import requests
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import Data
from mylib import *
from pylab import *
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling



#--------------
# data
#--------------
# path in linux
PATH_LINUX = '~/PycharmProjects/Place_Category_Prediction/data'
CHICAGO_ADJ_MATRIX = PATH_LINUX + '/feature_data/chicago_adj_matrix.csv'
CHICAGO_ADJ_MATRIX_WITH_DISTANCE = PATH_LINUX + '/feature_data/chicago_adj_matrix_with_distance.csv'
CHICAGO_EDGE_INDEX_PT = PATH_LINUX + '/pt_data/chicago_edge_index_tensor_50m.pt'
CHICAGO_X_PT = PATH_LINUX + '/pt_data/chicago_x_tensor_50m.pt'
CHICAGO_Y_PT = PATH_LINUX + '/pt_data/chicago_y_tensor_50m.pt'
CHICAGO_FEATURE_DATA = '~/PycharmProjects/Place_Category_Prediction/data/feature_data/chicago_dataset_prep.csv'

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
def construct_dataframe():
    edges = pd.read_csv(chicago_adj_matrix)
    edges.columns = ['edge_id', 'start_pid', 'end_pid']

    raw_data = pd.read_csv(chicago_node_feature_label)
    node_feature = raw_data['place_embedding']
    label_str_list = raw_data['label_encoding']

    start_node_list = edges['start_pid']
    end_node_list = edges['end_pid']

    raw_data['input_feature'] = node_feature.apply(lambda row: embedding_string_to_vector(row))
    raw_x = raw_data['input_feature']
    input_df_labels = ['pid_int', 'input_feature']
    raw_data[input_df_labels].to_csv(input_x_file)

    raw_data['label_vector'] = label_str_list.apply(lambda row: label_string_to_vector(row))
    raw_y = raw_data['label_vector']
    label_embedding_df_labels = ['pid_int', 'label_vector']
    raw_data[label_embedding_df_labels].to_csv(label_y_file)

    return start_node_list, end_node_list, raw_x, raw_y


def construct_dataframe_chicago_distance():
    edges = pd.read_csv(chicago_adj_matrix_with_distance)
    edges.columns = ['edge_id', 'start_pid', 'end_pid', 'distance']

    raw_data = pd.read_csv(chicago_node_feature_label)
    node_feature = raw_data['place_embedding']
    label_str_list = raw_data['label_encoding']

    start_node_list = edges['start_pid']
    end_node_list = edges['end_pid']

    raw_data['input_feature'] = node_feature.apply(lambda row: embedding_string_to_vector(row))
    raw_x = raw_data['input_feature']

    raw_data['label_vector'] = label_str_list.apply(lambda row: label_string_to_vector(row))
    raw_y = raw_data['label_vector']
    edge_distance = edges['distance']

    return start_node_list, end_node_list, raw_x, raw_y, edge_distance


def read_input_and_label():
    input_vector = pd.read_csv(input_x_file)
    x = input_vector['input_feature']

    label_vector = pd.read_csv(label_y_file)
    y = label_vector['label_vector']

    edges = pd.read_csv(chicago_adj_matrix)
    edges.columns = ['edge_id', 'start_pid', 'end_pid']

    start_node_list = edges['start_pid']
    end_node_list = edges['end_pid']

    return start_node_list, end_node_list, x, y



def visualize_loss(epoch_index, loss_vals):
    plot(epoch_index, loss_vals)
    xlabel('epoch')
    ylabel('loss')
    title('Loss')
    grid(True)
    show()
    plt.savefig('loss.png')


# save tensors
torch.save(edge_index, 'chicago_edge_index_tensor_50m.pt')
torch.save(x, 'chicago_x_tensor_50m.pt')
torch.save(y, 'chicago_y_tensor_50m.pt')
torch.save(dist, 'chicago_edge_distance_tensor_50m.pt')


# load Chicago tensors - x, y and edges
edge_index = torch.load('edge_index_tensor.pt')
x = torch.load('x_tensor.pt')
y = torch.load('y_tensor.pt')
dataset = Data(x=x, edge_index=edge_index, y=y)


#------------------------------------
# divide training and test dataset
#------------------------------------
num_nodes = dataset.num_nodes
idx = np.arange(num_nodes - 1)
np.random.shuffle(idx)
val_nums = int((num_nodes * 20) / 100)
train_nums = num_nodes - val_nums
val_idx = idx[:val_nums]
train_idx = idx[val_nums:]

dataset.train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
dataset.train_mask[train_idx] = 1  # train only on the 80% nodes
dataset.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool) # test on 20 % nodes
dataset.test_mask[val_idx] = 1


# -----------------------
# Build Network
#------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 11)
        self.linear1 = torch.nn.Linear(dataset.num_node_features, 16)
        self.linear2 = torch.nn.Linear(16, 11)
        self.conv3 = GCNConv(dataset.num_node_features, 126)
        self.conv4 = GCNConv(126, 11)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):  # as for the update part, the aggregated message and the current node embedding is aggregated, then it is multiplied by another weight matrix and applied another activation function
        # aggr_out has shape [N, out_channels]
        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding


class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = SAGEConv(2690582, 768)
        self.conv2 = SAGEConv(32, 16)
        self.lin1 = torch.nn.Linear(32, 11)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(62)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.lin1(x)

        return F.log_softmax(x, dim=1)


class Net4(torch.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()

        self.conv1 = GCNConv(dataset.num_node_features, 126)    # parameters - in_channel, out_channel, the other parameters are booleans
        self.conv2 = GCNConv(126, 62)
        self.linear1 = torch.nn.Linear(768, 11)
        self.linear2 = torch.nn.Linear(126, 11)
        self.pool1 = TopKPooling(126, ratio=0.5)
        self.pool2 = TopKPooling(62, ratio=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, _, _, _ = self.pool1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, _, _, _ = self.pool2(x, edge_index)

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
for epoch in range(1000):
    print('epoch : ', epoch)
    epoch_list.append(epoch)
    optimizer.zero_grad()
    out = model(data)
    # loss = criterion(torch.max(out, 1)[1], torch.max(data.y, 1)[1])  # index of max value in the tensor
    loss = criterion(out, torch.max(data.y, 1)[1])

    print('loss : ', loss)
    print('loss value : ', loss.item())
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()


#---------------------------
# visualize loss per epoch
#---------------------------
visualize_loss(epoch_list, loss_list)


#--------------------
# model evaluation
#--------------------
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
       print('wrong prediction POI_index :', val_idx[i])

print('correct : ', correct)
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))


#------------------
# save model
#------------------
saved_model = 'saved_model.pkl'
torch.save(model, saved_model)

