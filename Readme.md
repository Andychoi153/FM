# Factorization Machine for Creiteo Click prediction

## Setup

- dependency
  - numpy
  - pickle
  - json

- train.txt
  - criteo training data source
  - run.py 와 동일한 디렉토리에 존재 해야함


## ETL
- Feature
    - Numeric feature I_i
      - Numeric 의 경우 하나의 column 이 하나의 feature 값을 가진다.
      - if value > 2 -> ln(I_i) 로 정수 형 내림

    - Categorical feature
      - Categorical value 중 출현 횟수가 5 회 보다 낮은 경우 Null 과 동일하게 처리
      - Categorical 은 각 category 값의 유무가 하나의 feature 와 대응된다.

    - Example
 ```[1,5,8,2,...,49d68486] -> [1,2,2,2,...,0,0,0,1,0,0,0,0,...,0]```
 [1,5,8,2] 는 각각 [1,2,2,2] 에 대응되고, [49d68486] 는 [0,0,0,1,0,0,0,0,...,0] 에 대응

- Compressed Sparse Matrix
  - Categorical 변수의 변환 방식 때문에 matrix 는 sparse 한 속성을 가지고 있음
  - ptr, data, indices 를 기반으로 행렬을 압축 할 수 있음
  - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html

- Partitioning
  - 로컬 컴퓨터 메모리 bound 로 인해  partitioning 을 통해 학습 진행
  - 10만 라인 씩 나누어 총 459개의 파일로 분할
  - 1~300 파일 은 train set, 301~459 파일은 test set


## FactorizationMachine

## Train
- 각 파티셔닝 테이블을 기준으로 디렉토리 내에 iterating 학습
- gradient 설계 단에서 병렬 실행을 염두에 두지 않고 구현, iterating 학습만 지원 함


## Test
- parallel train
  - 동일한 파라미터를 기반으로 학습하기 때문에 병렬 연산 가능

- if prob > 0.5 then 1 else 0

## Loss function for Gradient
-  Field-aware Factorization Machines for CTR Prediction 에서 정의한 binary loss function 구현
    https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
- - t / (e^(t * fm(x)) + 1)


## Loss function for Evaluation
- -(t*log(p) + (1-t)log(1-p))
- p = sigmoid(fm(x))


## Result

- predict_result
  - Format
   - [<Line Number>,<Predicted Label>,<Predicted Probability>]

- update_params.pkl
  - trained numpy array [w0, v, w]

- Train log loss: 0.473070431740596
- Test log loss: 0.4727100371593292