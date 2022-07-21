from binaryninja import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product
from sklearn.decomposition import PCA
from  collections import Counter
import random
import os
import re
import pickle
import math

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
                    symbols[j] = "symbol"
                elif int(symbols[j], 16) in string_map:
                    symbols[j] = "string"
                else:
                    symbols[j] = "address"
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode]+operand)


def random_walk(g,length, symbol_map, string_map):
    sequence = []
    for n in g:
        if n != -1 and 'text' in g.node[n]:
            s = []
            l = 0
            s.append(parse_instruction(g.node[n]['text'], symbol_map, string_map))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    if 'text' in g.node[cur]:
                        s.append(parse_instruction(g.node[cur]['text'], symbol_map, string_map))
                        l += 1
                    else:
                        break
                else:
                    break
            sequence.append(s)
        if len(sequence) > 5000:
            print("early stop")
            return sequence[:5000]
    return sequence

def process_file(f, window_size):
    symbol_map = {}
    string_map = {}
    print(f)
    bv = BinaryViewType.get_view_of_file(f)
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    function_graphs = {}

    for func in bv.functions:
        G = nx.DiGraph()
        label_dict = {}   
        add_map = {}
        for block in func:
            # print(block.disassembly_text)
            curr = block.start
            predecessor = curr
            for inst in block:
                label_dict[curr] = bv.get_disassembly(curr)
                G.add_node(curr, text=bv.get_disassembly(curr))
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G    

    with open('cfg_train.txt', 'a') as w:
        for name, graph in function_graphs.items():
            sequence = random_walk(graph, 40, symbol_map, string_map)
            for s in sequence:
                if len(s) >= 4:
                    for idx in range(0, len(s)):
                        for i in range(1, window_size+1):
                            if idx - i > 0:
                                w.write(s[idx-i] +'\t' + s[idx]  + '\n')
                            if idx + i < len(s):
                                w.write(s[idx] +'\t' + s[idx+i]  + '\n')
    # gc.collect()

def main():
    bin_folder = '/path/to/binaries' 
    file_lst = []
    str_counter = Counter()
    window_size = 1;
    for parent, subdirs, files in os.walk(bin_folder):
        if files:
            for f in files:
                file_lst.append(os.path.join(parent,f))
    i=0
    for f in file_lst:
        print(i,'/', len(file_lst))
        process_file(f, window_size)
        i+=1

if __name__ == "__main__":
    main()