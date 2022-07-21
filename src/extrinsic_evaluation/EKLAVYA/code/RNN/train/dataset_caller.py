import pickle
import os
import numpy as np
from multiprocessing import Pool

embed_info = {}
type_info = {
    'char': 0,
    'int': 1,
    'float': 2,
    'pointer': 3,
    'enum': 4,
    'struct': 5,
    'union': 6
}


def approximate_type(type_str):
    int_list = ['_Bool', 'unsigned int', 'int', 'long long int', 'long long unsigned int', 'unsigned short',
                'short unsigned int', 'short', 'long unsigned int', 'short int', 'long int']
    char_list = ['char', 'unsigned char', 'signed char']
    if type_str[-1] == '*' or type_str == 'func_ptr' or type_str.split()[0][-1] == '*':
        return 'pointer'
    elif type_str in int_list:
        return 'int'
    elif type_str[:5] == 'enum ':
        return 'enum'
    elif type_str in char_list:
        return 'char'
    elif type_str[:7] == 'struct ':
        return 'struct'
    elif type_str[:6] == 'union ':
        return 'union'
    elif type_str == 'double' or type_str == 'long double':
        return 'float'
    else:
        return type_str


def one_hot_encoding(label_id, class_num):
    temp = np.zeros(class_num)
    temp[label_id] = 1
    return temp


def get_single_num_args(folder_path, file_name, func_list, embed_dim, max_length, class_num):
    file_path = os.path.join(folder_path, file_name)
    extract_info = {}
    with open(file_path) as f:
        file_info = pickle.load(f)
    for whole_func_name in func_list:
        '''callee_name#caller_name#indice'''
        temp = whole_func_name.split('#')
        callee_name = temp[0]
        caller_name = temp[1]
        indice = int(temp[2])
        func_tag = '%s#%s' % (file_name, whole_func_name)
        extract_info[func_tag] = {}
        # inst_bytes = file_info['functions'][caller_name]['inst_bytes'][:indice]
        temp_data = []
        indice_list = sorted(range(indice), reverse=True)
        for indice_id in indice_list:
            inst = file_info['functions'][caller_name]['inst_bytes'][indice_id]
            if str(inst) in embed_info:
                temp_data.append(embed_info[str(inst)]['vector'])
            else:
                temp_data.append([0.0] * embed_dim)
            if len(temp_data) >= max_length:
                break
        temp_data = np.asarray(temp_data)
        if temp_data.shape[0] < max_length:
            extract_info[func_tag]['length'] = temp_data.shape[0]
            temp_zero = np.zeros((max_length - temp_data.shape[0], embed_dim))
            temp_data = np.concatenate((temp_data, temp_zero), axis=0)
        else:
            extract_info[func_tag]['length'] = temp_data.shape[0]
        extract_info[func_tag]['data'] = temp_data
        extract_info[func_tag]['label'] = one_hot_encoding(file_info['functions'][callee_name]['num_args'], class_num)
    return extract_info


def get_single_args_type(folder_path, file_name, func_list, embed_dim, max_length, class_num, arg_no):
    file_path = os.path.join(folder_path, file_name)
    extract_info = {}
    with open(file_path) as f:
        file_info = pickle.load(f)
    for whole_func_name in func_list:
        '''callee_name#caller_name#indice'''
        temp = whole_func_name.split('#')
        callee_name = temp[0]
        caller_name = temp[1]
        indice = int(temp[2])
        func_tag = '%s#%s' % (file_name, whole_func_name)
        extract_info[func_tag] = {}
        # inst_bytes = file_info['functions'][caller_name]['inst_bytes'][:indice]
        temp_data = []
        indice_list = sorted(range(indice), reverse=True)
        for indice_id in indice_list:
            inst = file_info['functions'][caller_name]['inst_bytes'][indice_id]
            if str(inst) in embed_info:
                temp_data.append(embed_info[str(inst)]['vector'])
            else:
                temp_data.append([0.0] * embed_dim)
            if len(temp_data) >= max_length:
                break
        temp_data = np.asarray(temp_data)
        if temp_data.shape[0] < max_length:
            extract_info[func_tag]['length'] = temp_data.shape[0]
            temp_zero = np.zeros((max_length - temp_data.shape[0], embed_dim))
            temp_data = np.concatenate((temp_data, temp_zero), axis=0)
        else:
            extract_info[func_tag]['length'] = temp_data.shape[0]
        extract_info[func_tag]['data'] = temp_data
        temp_type = approximate_type(file_info['functions'][callee_name]['args_type'][arg_no])
        extract_info[func_tag]['label'] = one_hot_encoding(type_info[temp_type], class_num)
    return extract_info


class Dataset(object):
    def __init__(self, data_folder, func_path, embed_path, thread_num, embed_dim, max_length, class_num, tag):
        global embed_info

        self.data_folder = data_folder
        self.tag = tag #num_args or type#0
        if self.tag == 'num_args':
            pass
        else:
            self.arg_no = int(self.tag.split('#')[-1])
        self.thread_num = thread_num
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.class_num = class_num

        with open(func_path) as f:
            func_info = pickle.load(f)
        self.train_func_list = np.asarray(func_info['train'])
        self.train_num = len(self.train_func_list)
        print('Loaded train function information ... %s' % func_path)
        print('Train Function Number: %d' % self.train_num)

        with open(embed_path) as f:
            embed_info = pickle.load(f)
        print('Loaded embed information ... %s' % embed_path)

        self._index_in_epoch = 0
        self._complete_epochs = 0

    def get_batch_data(self, batch_func_list):
        func_list = sorted(batch_func_list)
        binary_name = ''
        input_func_list = []
        batch_info = {}
        pool = Pool(self.thread_num)
        if self.tag == 'num_args':
            for whole_func_name in func_list:
                if binary_name == '':
                    binary_name = whole_func_name.split('#')[0]
                    input_func_list.append('#'.join(whole_func_name.split('#')[1:]))
                else:
                    if binary_name == whole_func_name.split('#')[0]:
                        input_func_list.append('#'.join(whole_func_name.split('#')[1:]))
                    else:
                        pool.apply_async(
                            get_single_num_args,
                            args=(self.data_folder, binary_name, input_func_list, self.embed_dim, self.max_length,
                                  self.class_num),
                            callback=batch_info.update
                        )
                        binary_name = whole_func_name.split('#')[0]
                        input_func_list = ['#'.join(whole_func_name.split('#')[1:])]
            if len(input_func_list) == 0:
                pass
            else:
                pool.apply_async(
                    get_single_num_args,
                    args=(
                    self.data_folder, binary_name, input_func_list, self.embed_dim, self.max_length, self.class_num),
                    callback=batch_info.update
                )
        else:  # self.tag == 'type#0'
            for whole_func_name in func_list:
                if binary_name == '':
                    binary_name = whole_func_name.split('#')[0]
                    input_func_list.append('#'.join(whole_func_name.split('#')[1:]))
                else:
                    if binary_name == whole_func_name.split('#')[0]:
                        input_func_list.append('#'.join(whole_func_name.split('#')[1:]))
                    else:
                        pool.apply_async(
                            get_single_args_type,
                            args=(self.data_folder, binary_name, input_func_list, self.embed_dim, self.max_length,
                                  self.class_num, self.arg_no),
                            callback=batch_info.update
                        )
                        binary_name = whole_func_name.split('#')[0]
                        input_func_list = ['#'.join(whole_func_name.split('#')[1:])]
            if len(input_func_list) == 0:
                pass
            else:
                pool.apply_async(
                    get_single_args_type,
                    args=(
                    self.data_folder, binary_name, input_func_list, self.embed_dim, self.max_length, self.class_num,
                    self.arg_no),
                    callback=batch_info.update
                )
        pool.close()
        pool.join()
        new_batch_data = {
            'data': [],
            'label': [],
            'length': []
        }
        for full_func_name in batch_info:
            new_batch_data['data'].append(batch_info[full_func_name]['data'])
            new_batch_data['label'].append(batch_info[full_func_name]['label'])
            new_batch_data['length'].append(batch_info[full_func_name]['length'])
        batch_info = {
            'data': np.asarray(new_batch_data['data'], dtype=np.float32),
            'label': np.asarray(new_batch_data['label'], dtype=np.float32),
            'length': np.asarray(new_batch_data['length'], dtype=np.float32)
        }
        return batch_info

    def get_batch(self, batch_size):
        start = self._index_in_epoch
        # shuffle for the first round
        if self._complete_epochs == 0 and self._index_in_epoch == 0:
            perm0 = np.arange(self.train_num)
            np.random.shuffle(perm0)
            self.train_func_list = self.train_func_list[perm0]

        # go to the next epoch
        if start + batch_size > self.train_num:
            self._complete_epochs += 1
            rest_example_num = self.train_num - start
            rest_func_list = self.train_func_list[start:self.train_num]
            # shuffle for the new epoch
            perm = np.arange(self.train_num)
            np.random.shuffle(perm)
            self.train_func_list = self.train_func_list[perm]
            # start a new epoch
            start = 0
            self._index_in_epoch = batch_size - rest_example_num
            end = self._index_in_epoch
            new_func_list = self.train_func_list[start:end]
            func_list_batch = np.concatenate((rest_func_list, new_func_list), axis=0)
            train_batch = self.get_batch_data(func_list_batch)
            return train_batch
        else:  # process current epoch
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            func_list_batch = self.train_func_list[start:end]
            train_batch = self.get_batch_data(func_list_batch)
            print(train_batch)
            return train_batch