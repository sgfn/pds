import numpy as np
from operator import itemgetter
import os
import pickle
import string
import time


def create_dict(path, *args, dump=True, dump_path='', dump_name='w_dict.pickle',
                verbose=True, **kwargs):
    """Create a word dict encompassing all words found in files from given path."""

    tr = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    if verbose:
        start_time = time.time()
        print(f'Creating word dict for files from {path}:\n')

    words_dict = {}
    for file_name in os.listdir(path):
        full_path = path + file_name
        if os.path.isfile(full_path):
            with open(full_path, 'r') as f:
                if verbose: print(f'Processing {file_name}...', end='')

                try:
                    for line in f:
                        for w in line.translate(tr).lower().rstrip('\n').split():
                            words_dict[w] = 0
                except BaseException as e:
                    print(f'\tException: {e}, {type(e)}')
                else:
                    if verbose: print('done')

    words_amount = len(words_dict)
    if verbose:
        print(f'\nFinished indexing, found {words_amount} unique words.')

    for i, key in enumerate(words_dict.keys()):
        words_dict[key] = i

    if dump:
        dump_path += dump_name
        if verbose: print(f'Dumping data to {dump_path}...', end='')

        with open(dump_path, 'wb') as f_out:
            pickle.dump(words_dict, f_out)

    if verbose: print(f'done\nTotal time: {round(time.time()-start_time, 2)} s\n')
    
    if not dump: return words_dict


def index_files(path, *args, w_dict={}, load=True, load_path='w_dict.pickle',
                dump=True, dump_path='', dump_name='f_dict.pickle', verbose=True, **kwargs):
    """Create a dict with word vectors, according to w_dict, for each file from given path."""
    
    tr = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    if verbose:
        start_time = time.time()
        print(f'Indexing files from {path} using words dict {load_path if load else "passed as argument"}:\n')

    if load:
        if verbose: print(f'Loading dict {load_path}...', end='')
        with open(load_path, 'rb') as f:    w_dict = pickle.load(f)
        if verbose: print('done')

    files_dict = {}
    words_amount = len(w_dict)

    for file_name in os.listdir(path):
        full_path = path + file_name
        if os.path.isfile(full_path):
            with open(full_path, 'r') as f:
                if verbose: print(f'Processing {file_name}...', end='')
                v = np.zeros(words_amount)

                try:
                    for line in f:
                        for w in line.translate(tr).lower().rstrip('\n').split():
                            v[w_dict[w]] += 1
                except BaseException as e:
                    print(f'\tException: {e}, {type(e)}')
                else:
                    if verbose: print('done')
                
        files_dict[file_name] = v
    if verbose: print('\nFinished creating vectors.')

    if dump:
        dump_path += dump_name
        if verbose: print(f'Dumping data to {dump_path}...', end='')

        with open(dump_path, 'wb') as f_out:
            pickle.dump(files_dict, f_out)

        if verbose: print('done')

    if verbose: print(f'Total time: {round(time.time()-start_time, 2)} s\n')
    
    if not dump: return files_dict


def keyword_search(*args, w_dict={}, f_dict={}, load_w=True, w_path='w_dict.pickle',
                   load_f=True, f_path='f_dict.pickle', detailed=False, **kwargs):
    """Find closest match from f_dict to query."""

    if load_w:
        with open(w_path, 'rb') as f:    w_dict = pickle.load(f)
    if load_f:
        with open(f_path, 'rb') as f:    f_dict = pickle.load(f)

    kw_v = np.zeros(len(w_dict))
    for arg in args:
        if arg in w_dict:   kw_v[w_dict[arg]] += 1
    
    kw_v_len = 0
    for kw_x_i in kw_v:
        kw_v_len += kw_x_i*kw_x_i
    kw_v_len **= 1/2

    max_cosine_metric = -1
    res = '[no match]'

    if kw_v_len == 0:
        return res
    
    if detailed:    results = []

    for file_name, file_v in f_dict.items():
        dot_product = 0
        file_v_len = 0
        for i, file_x_i in enumerate(file_v):
            dot_product += kw_v[i] * file_x_i
            file_v_len += file_x_i*file_x_i
        file_v_len **= 1/2
    
        cosine_metric = dot_product / (kw_v_len*file_v_len)
        if detailed:    results.append((file_name, cosine_metric))
        if cosine_metric > max_cosine_metric:
            max_cosine_metric = cosine_metric
            res = file_name

    if detailed:
        counter = 5
        print('FILE\t\t\t% MATCH')
        for file_name, cosine_metric in sorted(results, reverse=True, key=itemgetter(1)):
            if cosine_metric > 0:
                if file_name[-4] == '.':    file_name = file_name[:-4]
                if len(file_name) < 20:
                    file_name += ' '* (20 - len(file_name))
                elif len(file_name) > 20:
                    t = file_name[:8]
                    file_name = t + '[...]' + file_name[-7:]
                print(f'{file_name}\t{round(cosine_metric*100, 4)}')
                counter -= 1
                if counter == 0:    break
        print()
    
    return res    


def kws_interactive(w_dict, f_dict):
    """Interactive command prompt mode for the keyword search feature."""

    tr = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    print("Keyword search - enter keywords separated by spaces\nEnter 'q' to quit, 'h' to display help message\n")
    query = input('Query: ')
    active = True
    detailed = False
    while active:
        if query == 'q':
            active = False
            break
        elif query == 'h':
            print('Available commands:\n\tq - quit\n\th - display this message')
            print('\td - toggle detailed mode\n\tn - input query with line breaks')
        elif query == 'd':
            detailed = not detailed
            print(f'Detailed output mode {"enabled" if detailed else "disabled"}')
        else:
            if query == 'n':
                print("Multiple line query. When done, type '$' and press enter")
                t = input()
                while t != '$':
                    query += t
                    t = input()
            res = keyword_search(*query.translate(tr).lower().split(), w_dict=w_dict, 
                                 f_dict=f_dict, load_w=False, load_f=False, detailed=detailed)
            print(f'Result: {res}')
        query = input('\nQuery: ')


if __name__ == '__main__':
    PATH='search_engine/archive/gutenberg/'
    DUMP_PATH='search_engine/data/'

    # create_dict(PATH, dump_path=DUMP_PATH)
    # index_files(PATH, load_path=DUMP_PATH+'w_dict.pickle', dump_path=DUMP_PATH)

    with open(DUMP_PATH+'w_dict.pickle', 'rb') as f:    w_dict = pickle.load(f)
    with open(DUMP_PATH+'f_dict.pickle', 'rb') as f:    f_dict = pickle.load(f)

    kws_interactive(w_dict, f_dict)
