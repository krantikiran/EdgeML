import numpy as np
from importlib import reload
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import sklearn.svm
sklearn.svm = reload(sklearn.svm)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample 
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import *
from scipy.spatial.distance import cdist 
from ctypes import c_uint,c_float 
from sklearn.gaussian_process.kernels import RBF
SCALE = 20
scale_factor = 1<<SCALE
f1 = open("svm_iter_digits.csv", "w")
f2 = open("svm_iter_iris.csv", "w")
f3 = open("svm_iter_wine.csv", "w")
def checkInputValidity32(xa, xb):
    assert len(xa.shape) > 1, "XA shape"
    assert len(xb.shape) > 1, "XB shape"
    assert xa.shape[1] == xb.shape[1], "Unequal number of colums"
    assert xa.dtype == np.int32
    assert xb.dtype == np.int32

def euclideanDist32(xa,xb):
    checkInputValidity32(xa,xb)
    xa = np.asarray(np.int64(xa))
    xb = np.asarray(np.int64(xb))
    result = np.ndarray(shape=(xa.shape[0],xb.shape[0]),dtype=np.int32)
    for i in range(xa.shape[0]):
        for j in range(xb.shape[0]):
            dotprod = (xa[i] - xb[j])
            assert dotprod.dtype == np.int64
            sqrprod = (dotprod * dotprod)>>(SCALE)
            assert sqrprod.dtype == np.int64
            prodsum = np.sum(sqrprod, dtype=np.int64)
            result[i][j] = np.int32(np.sqrt(prodsum) * (1<<(int(SCALE/2))))
    return (result)
    
# xa = np.asarray(np.int32([[1,2,3,4,5,6],[1,2,3,4,5,6]]))
# xb = np.asarray(np.int32([[9,10,11,12,13,14],[9,10,11,12,13,14]]))
# print(np.sum((xa[0]-xb[0])*(xa[0]-xb[0])),np.sqrt(np.sum((xa[0]-xb[0])*(xa[0]-xb[0]))))

# print(cdist(xa,xb))
# xb = xb * scale_factor
# xa = xa * scale_factor
# print(euclideanDist32(xa,xb)/scale_factor)
def fixedPointRound(elem):
    mask = np.int32(1<<(SCALE-1))
    val = np.int32(elem) & mask
    if val == 0:
        return np.int32(elem >> SCALE)
    else:
        return np.int32((elem >> SCALE) + 1)

def fixedExp(elem):
    try:
        elem = np.int32(elem)
    except:
        assert False, "FixedExp input conversion failed "

    t1 = np.int64()
    t2 = np.int64()
    # if elem < -20*scale_factor:
    #     return 0
    t1 = np.int64(elem) * np.int64(1.44269504089 * (1<<SCALE))
    N = np.int32(fixedPointRound(np.int32(t1 >> SCALE)))
    t2 = np.int64(N*scale_factor) * np.int64(0.69314718056 * (1<<SCALE))
    assert type(t2) == np.int64
    d = elem - np.int32(t2 >> SCALE)
    assert type(d) == np.int32

    y = d
    ans = scale_factor
    # count = 
    # for _ in range(itercount):
    #     big_y = np.int64(y)*np.int64(d)
    #     y = np.int32(big_y >> SCALE)
    #     y = np.int32(y/count)
    #     ans = ans + y
    #     count = count + 1

    if N > 0:
        ans = ans << N
    else:
        ans = ans >> (-1*N)
    
    return ans


def digits():
    data, target  =  load_digits(return_X_y=True)
    data = np.asarray(np.int32(data*scale_factor))
    target = np.asarray(np.int32(target*scale_factor))
    train_X, test_X, train_Y, test_Y = train_test_split(data, target, test_size=0.3, random_state = 0)
    assert train_X.dtype == np.int32
    assert train_Y.dtype == np.int32
    assert test_X.dtype == np.int32
    assert test_Y.dtype == np.int32
    # dists = cdist(np.asarray(np.int64(test_X)), np.asarray(np.int64(train_X)), metric='euclidean')
    dists = euclideanDist32(test_X, train_X)
    inp = dists>>1
    inp = -inp
    print(np.min(inp)/scale_factor, np.max(inp)/scale_factor)
    f = np.vectorize(fixedExp, otypes=[np.int32])
    res = f(inp)
    res_out = train_Y[np.argmax(res, axis = 1)]
    res_out = np.asarray(np.float32(res_out))
    score2 = accuracy_score(test_Y, res_out)
    print(score2)
    # f1.write(str(score2*100) + "\n")

def iris():
    data, target  =  load_iris(return_X_y=True)
    data = np.asarray(np.int32(data*scale_factor))
    target = np.asarray(np.int32(target*scale_factor))
    train_X, test_X, train_Y, test_Y = train_test_split(data, target, test_size=0.3, random_state = 0)
    assert train_X.dtype == np.int32
    assert train_Y.dtype == np.int32
    assert test_X.dtype == np.int32
    assert test_Y.dtype == np.int32
    # dists = cdist(np.asarray(np.int64(test_X)), np.asarray(np.int64(train_X)), metric='euclidean')
    dists = euclideanDist32(test_X, train_X)
    inp = dists>>1
    inp = -inp
    print(np.min(inp)/scale_factor, np.max(inp)/scale_factor)
    f = np.vectorize(fixedExp, otypes=[np.int32])
    res = f(inp)
    res_out = train_Y[np.argmax(res, axis = 1)]
    res_out = np.asarray(np.float32(res_out))
    score2 = accuracy_score(test_Y, res_out)
    print(score2)
    #f2.write(str(score2*100) + "\n")

def wine():
    data, target  =  load_wine(return_X_y=True)
    data = np.asarray(np.int32(data*scale_factor))
    target = np.asarray(np.int32(target*scale_factor))
    train_X, test_X, train_Y, test_Y = train_test_split(data, target, test_size=0.3, random_state = 0)
    assert train_X.dtype == np.int32
    assert train_Y.dtype == np.int32
    assert test_X.dtype == np.int32
    assert test_Y.dtype == np.int32
    # dists = cdist(np.asarray(np.int64(test_X)), np.asarray(np.int64(train_X)), metric='euclidean')
    dists = euclideanDist32(test_X, train_X)

    inp = dists>>1
    inp = -inp
    print(np.min(inp)/scale_factor, np.max(inp/scale_factor))
    f = np.vectorize(fixedExp, otypes=[np.int32])
    res = f(inp)
    res_out = train_Y[np.argmax(res, axis = 1)]
    res_out = np.asarray(np.float32(res_out))
    score2 = accuracy_score(test_Y, res_out)
    print(score2)
    #f3.write(str(score2*100) + "\n")

# for i in range(13):
#     global itercount
    # itercount = i
digits()
iris()
wine()

f1.close()
f2.close()
f3.close()