'''
Input: the whole dataset (in order to get the whole vocabulary)
Output:
output_path: the input for embedding model
error_path: save all the error information in (especially when two distinct instructions map to same integer)
int2insn_map_path: the map information(int -> insn (int list))
'''

import pickle
import argparse
import sys
import os
import insn_int


def get_file_path(folder_path, tag):
    path_list=[]
    file_list=os.listdir(folder_path)
    '''initial path list'''
    for file_name in file_list:
        path_list.append(os.path.join(folder_path, file_name))

    final_path_list=[]
    tag_len = len(tag)
    '''get all specific path'''
    while(len(path_list) > 0):
        source_path=path_list[0]
        path_list.remove(source_path)
        if not os.path.isdir(source_path) and (source_path[-tag_len-1] == '.') and (source_path[-tag_len:] == tag):
            final_path_list.append(source_path)
        elif os.path.isdir(source_path):
            file_list=os.listdir(source_path)
            for file_name in file_list:
                path_list.append(os.path.join(source_path, file_name))
        else:
            pass
    return final_path_list


class GetVocab(object):
    def __init__(self, config):
        self.config = config
        self.path_list=get_file_path(self.config['input_folder_path'], 'pkl')
        self.int2insn_map=dict()
        self.get_embed_input()

    def get_embed_input(self):
        cnt = 0
        for file_path in self.path_list:
            temp=pickle.load(open(file_path))
            insn2int_list = []
            for func_name in temp['functions']:
                for insn in temp['functions'][func_name]['inst_bytes']:
                    int_value=insn_int.insn2int_inverse(insn)
                    if int_value in self.int2insn_map:
                        if self.int2insn_map[int_value] != insn:
                            error_str='[ERROR] different insns map to same integer!!!!'
                            print(error_str)
                            with open(self.config['error_path'], 'a') as f:
                                f.write(error_str+'\n')
                                f.write('format: [int_value] insn1 # insn2\n')
                                f.write('[%d] %s # %s\n' % (int_value, str(self.int2insn_map[int_value]), str(insn)))
                    else:
                        self.int2insn_map[int_value]=insn
                    insn2int_list.append(str(int_value))
            with open(self.config['output_path'], 'ab') as f:
                if cnt == 0:
                    pass
                else:
                    f.write(' ')
                f.write(' '.join(insn2int_list))
            cnt+=1
            '''print the process'''
            if cnt % 100==0:
                print('[embed_input] Unpickle files: %d' % cnt)
            else:
                pass
        print('[embed_input] Got the input for training the embedding model!')
        with open(self.config['int2insn_map_path'], 'w') as f:
            pickle.dump(self.int2insn_map, f)
        print('[embed_input] Saved the integer-insn mapping information!')
        print('[embed_input] END!')


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder_path', dest='input_folder_path', help='The data folder saving binaries information.', type=str, required=True)
    parser.add_argument('-o', '--output_path', dest='output_path' ,help='The file saving the input for embedding model.', type=str, required=False, default='embed_input')
    parser.add_argument('-e', '--error_path', dest='error_path' ,help='The file saving all error information. ', type=str, required=False, default='error_log')
    parser.add_argument('-m', '--int2insn_map_path', dest='int2insn_map_path', help='The file saving the map information (int -> instruction (int list)).', type=str, required=False, default='int2insn.map')

    args = parser.parse_args()

    config_info = {
        'input_folder_path': args.input_folder_path,
        'output_path': args.output_path,
        'error_path': args.error_path,
        'int2insn_map_path': args.int2insn_map_path
    }
    return config_info

def main():
    config_info = get_config()
    my_vocab=GetVocab(config_info)

if __name__ == '__main__':
    main()