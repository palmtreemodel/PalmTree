import os
import pickle
import random
import time

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



def main():
    random.seed(time.time())
    splitFuncDict = {}
    train = []
    test = []
    path_list = get_file_path('smalldata/pickles','pkl')
    for file_path in path_list[0:10]:
        temp=pickle.load(open(file_path))
        for func_name in temp['functions']:
            (filepath,filename) = os.path.split(file_path)
            if random.random() >= 0.3:
                train.append(filename + '#' + func_name)
            else:
                test.append(filename + '#' + func_name)
    splitFuncDict['train'] = train
    splitFuncDict['test'] = test
    with open('outputs/split_func.pkl', 'wb') as f:
        pickle.dump(splitFuncDict, f)


if __name__ == '__main__':
    main()