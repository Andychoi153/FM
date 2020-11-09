import numpy as np


class FactorizationMachine:
    """
    클릭 예측(Classification)을 위한 Factorization Machine
    2-way variable interactions

    Parameters
    ---------
    latent_factors : int
        latent factors k

    learning_rate : float
        learning rate for gradient descent

    random_seed : int
        initialize v


    Attributes
    ---------
    bias
        bias
    v
        latent vector
    w
        linear coef

    """
    def __init__(self, latent_factors=10, learning_rate=0.1, random_seed=1, bias=0, w=None, v=None):
        self.latent_factors = latent_factors
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.loss = 0
        self.bias = bias
        self.w = w
        self.v = v

    def fit(self, x, y):
        """
        Fitting FM Classification model

        :param x: train x data set
        :param y: trin y data set
        """

        n_samples, n_features = x.shape

        if self.w is None:
            self.w = np.zeros(n_features)

        if self.v is None:
            self.v = np.random.normal(
                scale=1/self.latent_factors, size=(self.latent_factors, n_features))

        y[y == 0] = -1

        for i in range(n_samples):
            self.bias, self.w, self.v, loss = _descent(x.data, x.indptr, x.indices,
                               y, i, self.bias, self.w,
                               self.v, self.latent_factors,
                               self.learning_rate)
            self.loss += loss
        self.loss /= n_samples

        return self

    def predict_proba(self, x):
        """
        x feature 에 대한 Classification probability

        :param x: test set
        :return: [0,1] 의 클릭 확률
        """
        z = self.predict(x)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, x):
        """

        :param x:
        :return:
        """
        n_samples, n_features = x.shape
        y_h = [None]*n_samples
        for idx, i in enumerate(range(n_samples)):
            y_h_, _ = _predict_row(x.data, x.indptr,
                                        x.indices,
                                        self.bias,
                                        self.w,
                                        self.v,
                                        self.latent_factors, i)
            y_h[idx] = y_h_

        return np.array(y_h)


def _gradient(y_h, y, i):
    """
    Field-aware Factorization Machines for CTR Prediction 에서 정의한 binary loss function 구현
    https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf음

    :param y_h: float, FM model equation 결과 값
    :param y: target label
    :param i: row index
    :return: loss gradient
    """
    loss_gradient = -y[i] / (np.exp(y[i] * y_h) + 1.0)
    return loss_gradient


def _w_descent(indptr, w, indices, learning_rate, data, i, grad):
    """
    w descent 항

    :param data: sparse matrix 의 data
    :param indptr: sparse matrix 의 indptr
    :param indices: sparse matrix 의 indices
    :param w: linear weight
    :param learning_rate: float, gradient descent 의 학습 비율
    :param i: x 의 row index
    :param grad: loss function - chain rule 미분의 w0, w, v 공통 항
    :return:
        w: updated linear weight vector
    """
    for ptr in range(indptr[i], indptr[i + 1]):
        x_loc = indices[ptr]
        w[x_loc] -= learning_rate * (grad * data[ptr])
    return w


def _v_descent(latent_factors, indptr, v,  indices, learning_rate, data, i, grad, v_x_sum):
    """
    2-way latent vector descent 항
    O(kn) 시간 복잡도 따르는 공식 사용

    :param data: sparse matrix 의 data
    :param indptr: sparse matrix 의 indptr
    :param indices: sparse matrix 의 indices
    :param v: latent factor vectors
    :param latent_factors: int, latent factor k
    :param learning_rate: float, gradient descent 의 학습 비율
    :param i: x 의 row index
    :param grad: loss function - chain rule 미분의 w0, w, v 공통 항
    :param v_x_sum: v와 x의 내적 곱
    :return:
        v: updated latent vector
    """
    for f in range(latent_factors):
        for ptr in range(indptr[i], indptr[i + 1]):
            x_loc = indices[ptr]
            diff = v_x_sum[f] * data[ptr] - v[f, x_loc] * data[ptr]**2
            v_gradient = grad * diff
            v[f, x_loc] -= learning_rate * v_gradient
    return v


def _descent(data, indptr, indices, y, i,
                w0, w, v, latent_factors, learning_rate):
    """

    :param data: sparse matrix 의 data
    :param indptr: sparse matrix 의 indptr
    :param indices: sparse matrix 의 indices
    :param w0: bias
    :param w: linear weight
    :param v: latent factor vectors
    :param latent_factors: int, latent factor k
    :param learning_rate: float, gradient descent 의 학습 비율
    :param i: x 의 row index
    :return:
        updated bias, w, v
        loss: log_loss
    """

    y_h, v_x_sum = _predict_row(data, indptr, indices, w0, w, v, latent_factors, i)

    y_h_p = 1.0 / (1.0 + np.exp(-y_h))
    loss = log_loss(y_h_p, (y[i]+1)/2)

    common_gradient = _gradient(y_h, y, i)

    w0 -= learning_rate * common_gradient
    w = _w_descent(indptr, w, indices, learning_rate, data, i, common_gradient)
    v = _v_descent(latent_factors, indptr, v, indices, learning_rate, data, i, common_gradient, v_x_sum)

    return w0, w, v, loss


def _predict_row(data, indptr, indices, w0, w, v, latent_factors, i):
    """
    feature 데이터 셋 row i 에 대한 FM model equation 결과 값

    :param data: sparse matrix 의 data
    :param indptr: sparse matrix 의 indptr
    :param indices: sparse matrix 의 indices
    :param w0: bias
    :param w: linear weight
    :param v: latent factor vectors
    :param latent_factors: int, latent factor k
    :param i: x 의 row index
    :return:
        y_h: calculated FM model equation
        v_x_sum: latent 항 중 v와 x 내적 곱, descent 항에 재 사용 하기 위해 return
    """

    linear_term = _calculate_linear(indptr, indices, data, w, i)
    latent_term, v_x_sum = _calculate_latent(indptr, indices, data, latent_factors, v, i)

    y_h = w0 + linear_term + latent_term

    return y_h, v_x_sum


def _calculate_linear(indptr, indices, data, w, i):
    """
    linear 항 계산

    :param data: sparse matrix 의 data
    :param indptr: sparse matrix 의 indptr
    :param indices: sparse matrix 의 indices
    :param w: linear weight
    :param i: x 의 row index
    :return: calculate
    """
    linear_sum = 0
    for ptr in range(indptr[i], indptr[i + 1]):
        x_loc = indices[ptr]
        linear_sum += w[x_loc] * data[ptr]
    return linear_sum


def _calculate_latent(indptr, indices, data, latent_factors, v, i):
    """
    latent 항 계산

    :param data: sparse matrix 의 data
    :param indptr: sparse matrix 의 indptr
    :param indices: sparse matrix 의 indices
    :param v: latent factor vectors
    :param i: x 의 row index
    :return:
        latent_sum: calculated latent 항
        v_x_sum: latent 항 중 v와 x 내적 곱, descent 항에 재 사용 하기 위해 return
    """
    latent_sum = 0
    v_x_sum = np.zeros(latent_factors)
    v_x_squared_sum_ = np.zeros(latent_factors)

    for f in range(latent_factors):
        for ptr in range(indptr[i], indptr[i + 1]):
            x_loc = indices[ptr]
            v_x = v[f, x_loc] * data[ptr]
            v_x_sum[f] += v_x
            v_x_squared_sum_[f] += v_x * v_x
        latent_sum += 0.5 * (v_x_sum[f] * v_x_sum[f] - v_x_squared_sum_[f])
    return latent_sum, v_x_sum


def log_loss(y_h, y):
    """
    prediction y_h 의 log loss

    :param y_h: float, probability [0,1]
    :param y: float, target label {1,0}
    :return: log loss
    """
    return -np.sum(np.log(y_h)*y + np.log(1-y_h)*(1-y))
