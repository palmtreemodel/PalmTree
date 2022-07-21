from binaryninja import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product
from sklearn.decomposition import PCA
import random
import os
import re
import tqdm
import pickle
from  collections import Counter
from memory_profiler import profile
import gc


def parse_instruction(ins, symbol_map, string_map):
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                if int(symbols[j], 16) in symbol_map:
                    symbols[j] = "symbol" # function names
                elif int(symbols[j], 16) in string_map:
                    symbols[j] = "string" # constant strings
                else:
                    symbols[j] = "address" # addresses 
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode]+operand)


def random_walk(g,length, symbol_map, string_map):
    sequence = []
    for n in g:
        if n != -1 and g.node[n]['text'] != None:
            s = []
            l = 0
            s.append(parse_instruction(g.node[n]['text'], symbol_map, string_map))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    s.append(parse_instruction(g.node[cur]['text'], symbol_map, string_map))
                    l += 1
                else:
                    break
            sequence.append(s)
    return sequence



def process_file(f):
    symbol_map = {}
    string_map = {}
    print(f)
    bv = BinaryViewType.get_view_of_file(f)

    # encode strings
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    function_graphs = {}

    for func in bv.functions:
        G = nx.DiGraph()
        G.add_node(-1, text='entry_point')
        line = 0
        label_dict = {}
        label_dict[-1] = 'entry_point' 

        for block in func.mlil:
            for ins in block: 

                G.add_node(ins.address, text=bv.get_disassembly(ins.address))
                label_dict[ins.address] = bv.get_disassembly(ins.address)
                depd = []
                for var in ins.vars_read:
                    depd = [(func.mlil[i].address, ins.address) 
                            for i in func.mlil.get_var_definitions(var) 
                            if func.mlil[i].address != ins.address]
                for var in ins.vars_written:
                    depd += [(ins.address, func.mlil[i].address)
                            for i in func.mlil.get_var_uses(var)
                            if func.mlil[i].address != ins.address]
                if depd:
                    G.add_edges_from(depd)

        for node in G.nodes:
            if not G.in_degree(node):
                G.add_edge(-1, node)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G
    
    with open('dfg_train.txt', 'a') as w:
        for name, graph in function_graphs.items():
            sequence = random_walk(graph, 40, symbol_map, string_map)
            for s in sequence:
               if len(s) >= 2:
                    for idx in range(1, len(s)):
                        w.write(s[idx-1] +'\t' + s[idx] + '\n')
    gc.collect()


def process_string(f):
    str_lst = [] 
    bv = BinaryViewType.get_view_of_file(f)
    for sym in bv.get_symbols():
        str_lst.extend(re.findall('([0-9A-Za-z]+)', sym.full_name))
    return str_lst




def main():
    bin_folder = '/path/to/binaries'
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


