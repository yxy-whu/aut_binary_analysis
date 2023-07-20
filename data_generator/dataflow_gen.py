#22.7.16 created by yxy

import networkx as nx
import os
import tqdm 
import re

def parse_instruction(ins)
    ins = re.sub('\s+', ', ', ins, 1)
    ins = re.sub('(\{)|(\})', '', ins, 1)
    
    return ins
    
    
    
def process_file(f):
    with open("callstack_new.out", "r") as file:
        bool is_call = false
        bool is_ret = false
        call_addr = ''
        call_parent_addr = ''


        G = nx.DiGraph()
        G.add_node('0', func='entry_point')

        for line in file:
            ins_sequence = []

            str_line = line.strip()
            if str_line[0:3] == 'CALL':
                is_call = true
            if str_line[0:2] == 'RET':
                is_ret = true


            if is_call and str_line[1] == '=':
                is_call = false
            if is_ret and str_line[1] == '=':
                is_ret = false


            if str_line[0] == '#' and is_call:
                tmp = re.split(': ', str_line)
                #called function address
                if tmp[0] == '#now call addr':
                    call_addr = tmp[1]
                    G.add_node(call_addr)
                #call parent address
                if tmp[0] == '#call parent addr':
                    call_parent_addr = tmp[1]
                    G.nodes[call_parent_addr][ins] = ins_sequence
                    ins_sequence = []
                    G.add_edge(call_parent_addr, call_addr)

                #root size just depth
                if tmp[0] == '#root size':
                    depth = tmp[1]


            if str_line[0] == '#' and is_ret:
                tmp = re.split(': ', str_line)
                #ret for which call site
                if tmp[0] == '#ret target':
                    ret_target = tmp[1]
                    G.nodes[ret_target][ins] = ins_sequence
                    ins_sequence = G.node[call_parent_addr][ins]
                #function name
                if tmp[0] == '#function name':
                    func_name = tmp[1]
                    G.nodes[ret_target][func] = func_name
                    G.nodes[ret_target][end] = true
            
            #this is an instruction
            if str_line[0] == '[':
                ins_sequence.append(parse_instruction(str_line))
            

            if not(is_call) and not(is_ret):
                if str_line[0] == '[':
                    ins_sequence.append(parse_instruction(str_line))


def main():
    assmbly_folder = ''
    file_lst = []
    str_counter = Counter()
    for parent, subdirs, files in os.walk(bin_folder):
        if files:
            for f in files:
                file_lst.append(os.path.join(parent,f))
    for f in tqdm(file_lst):
        process_file(f)

if __name__ == "__main__":
    main()