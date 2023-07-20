from platform import node
import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product
from sklearn.decomposition import PCA
from  collections import Counter
from networkx.classes.reportviews import EdgeDataView
from registers import *
import random
import os
import re
import pickle
import math
import csv

def parse_instruction(ins):
    """
    parsing a instruction from [push rbp] to push rbp
    :param ins: instruction for string type 
    """
    # ins = re.sub('\s+', ', ', ins, 1)
    # ins = re.sub('(\[)|(\])', '', ins, 1)
    # ins = re.sub('(\[)|(\])', '', ins, 1)
    ins = re.sub('(\[)', '', ins, 1)
    ins = re.sub('(\])', '', ins, 1)
    
    return ins

def bin_to_list(source, target):
    """
    transform binary data to list 
    :param source: binary data for string type
    :param target: the length of store binary data list
    """
    S = [0] * target
    count = 0
    
    for i in source:
        S[count] = int(i)
        count += 1
    return S

def get_dict_key(dic, value):
    """
    finding the key corresponding to the value
    :param dic: target dictory 
    :param value: target value 
    """
    keys = list(dic.keys())
    values = list(dic.values())
    try:
        idx = values.index(value)
    except ValueError:
        return 0

    # if keys.count(idx) == 0:
    #     return 0
    # else:
    #     return keys[idx]
    return keys[idx]

def write_txt(sequence, window_size, file_txt):
    """
    wirte window_size instruction sequence in file as data set
    :param sequence: input instruction sequence 
    :param window_size: training model adpot predicte window_size
    """
    if len(sequence)>= 4:
        for idx in range(0, len(sequence)):
            for i in range(1, window_size+1):
                if idx - i > 0:
                    file_txt.write(sequence[idx-i] +'\t' + sequence[idx]  + '\n')
                if idx + i < len(sequence):
                    file_txt.write(sequence[idx] +'\t' + sequence[idx+i]  + '\n')

def process_file(f):
    """
    trandform trace record into three file: call_graph, csv, instruction sequence
    csv for GAT model
    instruction sequence for bert model

    :param f: trace record file path 
    """
    #get originaldata1.out
    tmp = re.split('/', f)
    #get originaldata1
    key_word = re.split('\\.', tmp[len(tmp) - 1])
    #change originaldata1.out to originaldata1.csv
    file_name = key_word[len(key_word) - 2] + '.csv'
    #change file path
    file_name = f.replace(tmp[len(tmp) - 2] + '/' +  tmp[len(tmp) - 1], 'call_graph_csv/'+file_name)

    #file_assembly is the data set for condtructing pre-training model
    file_assembly = key_word[len(key_word) - 2] + '.txt'
    file_assembly = f.replace(tmp[len(tmp) - 2] + '/' +  tmp[len(tmp) - 1], 'training/'+file_assembly)
    window_size = 1


    with open(f, "r") as file, open(file_name, "w") as file_csv, open(file_assembly, "w") as file_txt:
        is_call = False
        is_ret = False
        call_addr = ''
        call_parent_addr = ''
        w_csv = csv.writer(file_csv)
        w_csv.writerow(["node_id", "node_feature", "head_node", "tail_node", "edge_feature"])
        node_id = 1
        node_dic = {}
        #recorde call relationship before ret
        before_ret_head = 0
        before_ret_tail = 0
        #recorde call function stack
        call_relationship = []
        test_call_relationship = []



        G = nx.DiGraph()
        
        #each node have five attributes
        G.add_node('0', ins_list = [], func_name = 'start_entry', depth = 0, end = True, parent = 0)
        ins_sequence = []

        for line in file:
            str_line = line.strip()
            #ins_sequence = []

            #recode an instruction
            if str_line[0] == '[':
                ins_sequence.append(parse_instruction(str_line))
            
            if str_line[0:4] == 'CALL':
                is_call = False
                is_ret = False

                is_call = True
            if str_line[0:3] == 'RET':
                is_call = False
                is_ret = False

                is_ret = True


            if str_line[0] == '#' and is_call:
                tmp = re.split(': ', str_line)
                #called function address
                if tmp[0] == '#now call addr':
                    call_addr = tmp[1]
                    G.add_node(call_addr, ins_list = ins_sequence, func_name = '', depth = 0, end = False, parent = 0)

                #call parent address
                if tmp[0] == '#call parent addr':
                    call_parent_addr = tmp[1]
                    if call_parent_addr == '0':
                        G.nodes['0']['ins_list'] = ins_sequence
                        # write data set for semantic learning in file.txt
                        write_txt(ins_sequence, window_size, file_txt)

                        ins_sequence = []
                    else:
                        G.nodes[call_parent_addr]['ins_list'] = ins_sequence

                        # write data set for semantic learning in file.txt
                        write_txt(ins_sequence, window_size, file_txt)

                        ins_sequence = []
                    G.nodes[call_addr]['parent'] = call_parent_addr 
                    
                    G.add_edge(call_parent_addr, call_addr)

                #root size just depth
                if tmp[0] == '#root size':
                    G.nodes[call_addr]['depth'] = tmp[0]


            if str_line[0] == '=' and is_call:
                #add data in csv file
                if call_parent_addr == '0':
                    tmp_call = call_addr[-8:]
                    head_node_feature = [0]*32
                    tail_node_feature = bin_to_list(bin(int(tmp_call, 16))[2:], 32)
                    tail_dict_key = get_dict_key(node_dic, tail_node_feature)
                    head_dict_key = get_dict_key(node_dic, head_node_feature)
                    
                    if tail_dict_key != 0:         
                        w_csv.writerow([-1, tail_node_feature, 0, tail_dict_key, 1])
                        before_ret_head = 0
                        before_ret_tail = tail_dict_key
                    
                    else:
                        node_dic[node_id] = tail_node_feature
                        w_csv.writerow([node_id, tail_node_feature, 0, node_id, 1])
                        before_ret_head = 0
                        before_ret_tail = node_id
                        node_id += 1
                    call_relationship.append(['0', call_addr])


                    #for test
                    test_call_relationship.append([before_ret_head, before_ret_tail])  





                else:
                    tmp_call_parent = call_parent_addr[-8:]
                    head_node_feature = bin_to_list(bin(int(tmp_call_parent, 16))[2:], 32)

                    tmp_call = call_addr[-8:]
                    tail_node_feature = bin_to_list(bin(int(tmp_call, 16))[2:], 32)

                    if tail_dict_key != 0:         
                        w_csv.writerow([-1, tail_node_feature, head_dict_key, tail_dict_key, 1])
                        before_ret_head = head_dict_key
                        before_ret_tail = tail_dict_key
                    else:
                        node_dic[node_id] = tail_node_feature
                        w_csv.writerow([node_id, tail_node_feature, head_dict_key, node_id, 1])
                        before_ret_head = head_dict_key
                        before_ret_tail = node_id
                        node_id += 1

                    call_relationship.append([call_parent_addr, call_addr])


                    #for test
                    test_call_relationship.append([before_ret_head, before_ret_tail])
                    
                


            if str_line[0] == '#' and is_ret:
                tmp = re.split(': ', str_line)
                #ret for which call site
                if tmp[0] == '#ret target':
                    #get a function ret and make last process
                    ret_target = tmp[1]
                    G.nodes[ret_target]['ins_list'] = ins_sequence

                    # write data set for semantic learning in file.txt
                    write_txt(ins_sequence, window_size, file_txt)

                    G.nodes[ret_target]['end'] = True
                    #init ins_sequence and node ID as depth-1 level for next process
                    ins_sequence = []
                    call_addr = G.nodes[ret_target]['parent']
                    call_parent_addr = G.nodes[call_addr]['parent']
                    ins_sequence = G.nodes[call_addr]['ins_list']


                #function name
                if tmp[0] == '#function name':
                    G.nodes[ret_target]['func_name'] = tmp[1]
            
            if str_line[0] == '=' and is_ret:
                #add data in csv file
                tmp_ret = ret_target[-8:]
                ret_feature = bin_to_list(bin(int(tmp_ret, 16))[2:], 32)

                if call_relationship[-1][1] == ret_target:
                    w_csv.writerow([-1, ret_feature, before_ret_tail, before_ret_head, 0])
                    call_relationship.pop()
                    

                    #for test
                    test_call_relationship.append([before_ret_tail, before_ret_head])



                else:
                    print('worry function call relationship')

                
                
    file.close()
    file_csv.close()

    # plt.figure(figsize=(10, 10))
    # pos = nx.random_layout(G)
    # nx.draw(G, pos=pos, with_labels=True)
    # plt.savefig("/home/func_logic/figure/figure_network", dpi=1000, bbox_inches='tight')

    


def main():
    bin_folder = '/home/xiaoyu_yi/func_logic/data/original_data' 
    file_lst = []
    #str_counter = Counter()
    #window_size = 10
    for parent, subdirs, files in os.walk(bin_folder):
        if files:
            for f in files:
                file_lst.append(os.path.join(parent,f))
    i=0
    for f in file_lst:
        print(i,'/', len(file_lst))
        process_file(f)
        i+=1

    

if __name__ == "__main__":
    main()