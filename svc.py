import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.externals import joblib
import time

if __name__ =="__main__":
    train_num = 5000
    test_num = 7000
    data = pd.read_csv('train.csv')
    train_data = data.values[0:train_num,1:]
    train_label = data.values[0:train_num,0]
    test_data = data.values[train_num:test_num,1:]
    test_label = data.values[train_num:test_num,0]
    t = time.time()

    #PCA降维
    pca = PCA(n_components=0.8, whiten=True)
    print('start pca...')
    train_x = pca.fit_transform(train_data)
    test_x = pca.transform(test_data)
    print(train_x.shape)

    # svm训练
    print('start svc...')
    svc = svm.SVC(kernel = 'rbf', C = 10)
    svc.fit(train_x,train_label)
    pre = svc.predict(test_x)

    #保存模型
    joblib.dump(svc, 'model.m')
    joblib.dump(pca, 'pca.m')

    # 计算准确率
    score = svc.score(test_x, test_label)
    print(u'准确率：%f,花费时间：%.2fs' % (score, time.time() - t))




