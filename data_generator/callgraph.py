import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pandas as pd
import torch
import os
import math

# class Data(object):
#     def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
#         self.x = x
#         self.edge_index = edge_index
#         self.edge_attr = edge_attr
#         self.y = y
 
#         for key, item in kwargs.items():
#             if key == 'num_nodes':
#                 self.__num_nodes__ = item
#             else:
#                 self[key] = item


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels_1):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(dataset.num_features, hidden_channels_1)
        self.conv2 = GATConv(hidden_channels_1, 10)
        #self.conv3 = GATConv(hidden_channels_2, 9)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv3(x, edge_index, edge_attr)
        return x




def load_col_csv(path, index_col):
    df = pd.read_csv(path)
    
    col = [row for row in df[index_col]]
    #del col[0]
    return col

def load_node_csv(path):
    df = pd.read_csv(path)
    #ls_i = []

    node_feature = [[]]
    #mapping = {i: index for i, index in enumerate(df.index.unique())}
    for id, feature in zip(df['node_id'], df['node_feature']):
        if id != -1:
            for i in feature:
                if i.isdigit():
                    node_feature[len(node_feature) - 1].append(int(i))
                    #ls_i.append(int(i))
            node_feature.append([])

    node_feature[len(node_feature) - 1] = [0]*32
    x = torch.tensor(node_feature, dtype=torch.float)
    return x

def load_node_lable(path):
    df = pd.read_csv(path)
    tmp = -1
    y = []
    record = []
    record_to_binary = []

    #find the most big node_id len
    for id in df['node_id']:
        if id > tmp:
            tmp = id
    tmp = len(bin(tmp)) - 1

    for i in range(tmp):
        record.append(pow(2, i))
        record_to_binary.append([0]*tmp)
        record_to_binary[len(record_to_binary)-1][tmp-1-i] = 1
    
    for id_real in df['node_id']:
        if id_real != -1:

            for record_count in range(len(record)):
                if record[record_count] < id_real:
                    continue

                if record[record_count] == id_real:
                    y.append(record_to_binary[record_count])
                    break

                if record[record_count] > id_real:
                    if record[record_count]-id_real > id_real-record[record_count-1]:
                        y.append(record_to_binary[record_count-1])
                    else:
                        y.append(record_to_binary[record_count])
                    break

            # for i in reversed(bin(id_real)[2:]):
            #     y[len(y)-1][node_len-1] = int(i)
            #     node_len -= 1
    
    y.append([0]*tmp)
    y[-1][tmp-1] = 1
    y = torch.tensor(y, dtype=torch.float)
    return y

def load_edge_csv(path, src, dst):
    src = load_col_csv(path, 'head_node')
    #print('src length: ', len(src))
    #print(src[0])
    # print(src)
    # for src_i in src:
    #     src[src_i-1] = int(src_i)

    dst = load_col_csv(path, 'tail_node')
    #print('dst length: ', len(dst))
    #print(dst[0])
    # for src_i in src:
    #     src[src_i-1] = int(src_i)

    edge_index = torch.tensor([src, dst])

    return edge_index


def load_edge_attr(path):
    df = pd.read_csv(path)

    #tmp_edge_feature = []
    edge_feature = []
    for y in df['edge_feature']:
        if y == 0:
            #tmp_edge_feature.append(1)
            #tmp_edge_feature.append(0)
            edge_feature.append([1, 0])
        else:
            #tmp_edge_feature.append(0)
            #tmp_edge_feature.append(1)
            edge_feature.append([0, 1])
        
    
    edge_attr = torch.tensor(edge_feature)
    return edge_attr

# def list_to_lable(y, len_y):
#     src_y = [0]*len_y
#     tmp_y = y.tolist()
#     record = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

#     iter_count = 0
#     for i in tmp_y:
#         count = 0   
#         for j in i:
#             if int(j + 0.5) >= 1:
#                 src_y[iter_count] += pow(2, count)
#             count += 1
#         iter_count += 1    

#     for y_count in range(len(src_y)):

#         for record_count in range(len(record)):
#             if record[record_count] < src_y[y_count]:
#                 continue

#             if record[record_count] == src_y[y_count]:
#                 src_y[y_count] = record_count
#                 continue

#             if record[record_count] > src_y[y_count]:
#                 if record[record_count]-src_y[y_count] > src_y[y_count]-record[record_count-1]:
#                     src_y[y_count] = record_count - 1
#                 else:
#                     src_y[y_count] = record_count
#                 continue

#     dst_y = torch.tensor(src_y)
#     return dst_y

def list_to_lable(y):
    tmp_y = y.tolist()
    record = []
    dst_y = []

    for i in range(len(tmp_y[0])):
        record.append(pow(2, i))
    
    for j in tmp_y:
        tmp_count = j.index(1.0)
        #dst_y.append(record[class_num-1-tmp_count])
        dst_y.append(tmp_count)

    dst_y = torch.tensor(dst_y)
    return dst_y



def train():
    GAT_model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = GAT_model(dataset.x, dataset.edge_index, dataset.edge_attr)  # Perform a single forward pass.

    #test for next stentce
    print(out.size())
    print(out)
    # print(dataset.y.size())
    # print(dataset.y)
    # print(out[dataset.y].size())
    # print(dataset.y[dataset.y].size())

    loss = criterion(out, dataset.y)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test():
    GAT_model.eval()
    out = GAT_model(dataset.x, dataset.edge_index, dataset.edge_attr)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    #test
    #print('pred size: ', pred.size())
    #print(pred)

    result = list_to_lable(dataset.y).to(device)

    test_correct = pred == result  # Check against ground-truth labels.
    # test_acc = int(test_correct.sum()) / int(dataset.y.sum())  # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(dataset.y.shape[0])
    return test_acc


def test_graph():
    GAT_model.eval()
    out = GAT_model(dataset.x, dataset.edge_index, dataset.edge_attr)
    # pred = out.argmax(dim=1)  # Use the class with highest probability.

    result = list_to_lable(dataset.y).to(device)
    result = result.tolist()

    #test
    #print('pred size: ', pred.size())
    #print(pred)

    # result = list_to_lable(dataset.y).to(device)

    # test_correct = pred == result  # Check against ground-truth labels.
    # # test_acc = int(test_correct.sum()) / int(dataset.y.sum())  # Derive ratio of correct predictions.
    # test_acc = int(test_correct.sum()) / int(dataset.y.shape[0])

    visualize(out, color=result)



def visualize(out, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()
    plt.savefig("test_50k.png")






# start 
csv_path = '/home/xiaoyu_yi/func_logic/data/call_graph_csv'
file_lst = []

for parent, subdirs, files in os.walk(csv_path):
    if files:
        for f in files:
            file_lst.append(os.path.join(parent,f))

i = 0
for f in file_lst:
    print(i, '/', len(file_lst))
    src_mapping = []
    dst_mapping = []
    x = load_node_csv(f)
    edge_index = load_edge_csv(f, src_mapping, dst_mapping)
    edge_attr = load_edge_attr(f)
    y = load_node_lable(f)
    dataset = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)

    #test for out of bound bug
    #print('dataset.num_features: ',dataset.num_features)
    print('data: ', dataset)
    #print(dataset.edge_index.max())
    #print(dataset.num_nodes)
    #assert dataset.edge_index.max() < dataset.num_nodes

    #first get net struction 
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    GAT_model = GAT(hidden_channels_1=250).to(device)
    dataset = dataset.to(device)
    optimizer = torch.optim.Adam(GAT_model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0003)
    criterion = torch.nn.CrossEntropyLoss()

    #second for training
    for epoch in range(1, 10):
        loss = train()
        if(epoch % 10000 == 0):
            print('========================================================')
            print('---->',f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if(epoch % 10000 == 0):
            #for testing
            test_acc = test()
            print(f'Test Accuracy: {test_acc:.4f}')
    
    test_graph()


    

def main():
    print('over')

if __name__ == "__main__":
    main()