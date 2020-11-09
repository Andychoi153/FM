import numpy as np


class crs_matrix_criteo:
    """
    크리테오 데이터 셋을 Compressed Sparse Row matrix 로 변환하는 함수

    Parameters
    ----------
    criteo_data : numpy array
        criteo 데이터 source

    numeric_cols : int
        데이터 feature 중 numeric cols 개수

    category_cells : dict default None
        key: category hash map
        value: integer labeled
        example:
            {"25c83c98:1, "ac5d8d16":2 ... ,"0b153874":N}

    Attributes
    ---------
    indptr: numpy array
        하나의 element 와 하나의 row 가 mapping 되어 indices, data 주소 포인터를 가지고 있음

    indices: numpy array
        indptr 과 결합되어 사용되는 row 의 feature column index 정보

    data: numpy array
        indptr, indices  결합되어 사용되는 row 의 feature column 의 value 정보

    """
    def __init__(self, criteo_data, numeric_cols=13, category_cells=None):
        self.indptr = [None] * criteo_data.shape[0]
        self.criteo_data = criteo_data
        self.indices = []
        self.data = []
        self.category_cells = category_cells
        self.n_cols_num = numeric_cols
        self.__convert()
        self.shape = (len(criteo_data), self.n_cols_num + len(self.category_cells))

    def __convert(self):
        """

        :return:
        """
        for ptr, d in enumerate(self.criteo_data):
            for idx, term in enumerate(d):
                if term:
                    if idx > self.n_cols_num-1:
                        index = self.category_cells.get(term)
                        if index:
                            # category number int label 첫 시작이 0(dummy) 시작, 성능 안나와서 폐기
                            # label 첫 시작이 1로 바뀌면서 index 최솟값도 0 에서 1로 변경
                            # 코드 처음 설계는 index 0 start
                            # 마지막 category code d12c1aad 가 train set 에서는 없다가 test set 등장, index out of range error 일으킴
                            # 코드 수정하기에는 training 시간이 촉박..
                            # todo category label 0 시작으로 변경 & 아래 if 구문 정리
                            if term != 'd12c1aad':
                                self.indices.append(self.n_cols_num + index)
                                self.data.append(1)

                        # else:
                        #     index = self.category_cells.get('dummy')
                        #     self.indices.append(self.n_cols_num + index)
                        #     self.data.append(1)

                    else:
                        term = float(term)
                        self.indices.append(idx)
                        if term > 2:
                            term = np.floor(np.log(term))
                        self.data.append(term)
            self.indptr[ptr] = len(self.indices)
        self.indptr.insert(0, 0)
