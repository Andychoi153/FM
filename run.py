import numpy as np
import json
from FM import FactorizationMachine, log_loss
from crs_matrix_criteo import crs_matrix_criteo
import multiprocessing
import pickle
import time


def iter_learn(file_name):
    # iterating learning
    from numpy import genfromtxt
    start = time.time()
    train = genfromtxt(file_name, dtype='str', delimiter='\t')
    train = train.T
    train_x = train[1:].T

    with open('category.json', 'r') as fp:
        category_cells = json.load(fp)

    try:
        with open('update_params.pkl', 'rb') as f:
            params = pickle.load(f)
        bias = params[0]
        w = params[1]
        v = params[2]

    except Exception:
        bias = 0
        w = None
        v = None

    x_train = crs_matrix_criteo(train_x, category_cells=category_cells)
    y_train = train[0].T.astype(np.int32)

    fm = FactorizationMachine(latent_factors=5, learning_rate=0.001, bias=bias, w=w, v=v)
    fm.fit(x_train, y_train)

    params = [fm.bias, fm.w, fm.v, fm.loss]

    with open('update_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    with open('param_history_{}.pkl'.format(file_name[:-4]), 'wb') as f:
        pickle.dump(params, f)

    print(fm.loss)
    print(time.time()-start)


def parallel_prediction(file_name):
    start = time.time()

    from numpy import genfromtxt
    test = genfromtxt(file_name, dtype='str', delimiter='\t')
    test = test.T
    test_x = test[1:].T

    with open('category.json', 'r') as fp:
        category_cells = json.load(fp)

    x_test = crs_matrix_criteo(test_x, category_cells=category_cells)
    y_test = test[0].T.astype(np.int32)

    with open('update_params.pkl', 'rb') as f:
        params = pickle.load(f)

    bias = params[0]
    w = params[1]
    v = params[2]

    fm = FactorizationMachine(latent_factors=5,
                              bias=bias, w=w, v=v)

    y_pred_prob = fm.predict_proba(x_test)
    loss = log_loss(y_pred_prob, y_test)
    loss /= len(y_test)

    with open('loss_{}.pkl'.format(file_name[:-4]), 'wb') as f:
        pickle.dump(loss, f)

    y_predict_labeld = y_pred_prob.copy()

    y_predict_labeld[y_predict_labeld >= 0.5] = 1
    y_predict_labeld[y_predict_labeld < 0.5] = 0

    result = np.array([y_pred_prob, y_predict_labeld, y_test])
    np.savetxt('test_result_{}.txt'.format(file_name[:-4]), result.T, delimiter='\t')

    print(loss)
    print(time.time()-start)
    return result


def partitioning_train_set():
    """
    train.txt 파일 100000 line 단위 chunk file 로 partitioning
    """
    lines_per_file = 100000
    chunk_file = None
    with open('train.txt') as source_file:
        for lineno, line in enumerate(source_file):
            if lineno % lines_per_file == 0:
                if chunk_file:
                    chunk_file.close()
                chunk_file_name = 'train_{}.txt'.format(lineno + lines_per_file)
                chunk_file = open(chunk_file_name, "w")
            chunk_file.write(line)
        if chunk_file:
            chunk_file.close()


def create_category_json():
    """
    category 문자 -> int label 매핑 json 모델 생성
    """
    from numpy import genfromtxt
    category_cells = {}
    file_ptrs = 460
    cnt = 0

    for i in range(1, file_ptrs):
        chunk = genfromtxt('train_{}.txt'.format(i * 100000), dtype='str', delimiter='\t')
        chunk = chunk.T
        category = np.unique(chunk[14:])

        for idx, term in enumerate(category):
            if term is not None:
                if type(term) is np.str_ and term != '':
                    val = category_cells.get(term)
                    if val is not None:
                        category_cells[term] = category_cells.get(term) + 1
                    else:
                        category_cells[term] = 0

    category_dict = dict()
    for (k, v) in category_cells.items():
        if v > 5:
            cnt += 1
            category_dict[k] = cnt


if __name__ == '__main__':
    # create `train_{k}.txt`
    # partitioning_train_set()

    # create `category.json`
    # category_json_create()

    train_file_ptr_start = 1
    train_file_ptr_end = 301

    test_file_ptr_start = 301
    test_file_ptr_end = 460
    iter_learn('train_100000.txt')
    ########################### train ##################################
    # 약 4 시간
    # file_names = [None] * (train_file_ptr_end - train_file_ptr_start)
    #
    # for idx, i in enumerate(range(train_file_ptr_start, train_file_ptr_end)):
    #     file_names[idx] = 'train_{}.txt'.format(i * 100000)
    #
    # for file_name in file_names:
    #     iter_learn(file_name)
    ###################################################################

    ########################### test ##################################
    # 약 10 분
    # file_names = [None] * (test_file_ptr_end-test_file_ptr_start)
    #
    # for idx, i in enumerate(range(test_file_ptr_start, test_file_ptr_end)):
    #     file_names[idx] = 'train_{}.txt'.format(i * 100000)
    #
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()*2)
    # pool.map(parallel_prediction, file_names)
    # pool.close()
    # pool.join()
    ###################################################################

    # concat & aggregation values
    start = 301
    end = 460
    length = 100000

    for i in range(301, 459):
        index = range((i - 1) * length + 1, i * length + 1)
        val = np.loadtxt('test_result_train_{}.txt'.format(i * length))
        proba = val.T[0]
        label = val.T[1]
        result = np.array([index, proba, label])
        with open("predict_result.txt", "ab") as f:
            np.savetxt(f, result.T)

    i = 459
    val = np.loadtxt('test_result_train_{}.txt'.format(i*length))
    index = range((i - 1) * length + 1, (i - 1) * length + len(val.T[0]) + 1)
    proba = val.T[0]
    label = val.T[1]
    result = np.array([index, proba, label])
    with open("predict_result.txt", "ab") as f:
        np.savetxt(f, result.T)

    train_param_agg = [None] * (train_file_ptr_end - train_file_ptr_start)

    for idx, i in enumerate(range(train_file_ptr_start, train_file_ptr_end)):
        with open('param_history_train_{}.pkl'.format(i * 100000), 'rb') as f:
            params = pickle.load(f)
            train_param_agg[idx] = params[-1]

    test_param_agg = [None] * (test_file_ptr_end - test_file_ptr_start)

    for idx, i in enumerate(range(test_file_ptr_start, test_file_ptr_end)):
        with open('loss_train_{}.pkl'.format(i * 100000), 'rb') as f:
            param = pickle.load(f)
            test_param_agg[idx] = param

    log_loss_test = sum(test_param_agg) / len(test_param_agg)
    log_loss_train = sum(train_param_agg) / len(train_param_agg)

    # log_loss_train: 0.473070431740596
    print(log_loss_train)

    # log_loss_test: 0.4727100371593292
    print(log_loss_test)
