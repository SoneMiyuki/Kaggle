from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import csv

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.


mlp_hw = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [100,100],activation = 'relu',alpha = 1e-5,random_state = 0)
mlp_hw.fit(X,y)
print('score : {:.2f}%'.format(mlp_hw.score(X,y)*100))

X_test = []
y_test = []

with open(r"test.csv",newline = '')  as f:
    reader = csv.reader(f)
    header_exist = True
    for row in reader: 
        if(header_exist):
            header_exist = False
            continue
        X_test.append([ float(x)/255. for x in row[0:784] ])
    y_test =     mlp_hw.predict(X_test)

with open(r'result.csv','w',newline='')as f:
    wt = csv.writer(f)
    headers = ['ImageId','Label']
    wt.writerow(headers)
    for i in range(0,len(y_test)):
        wt.writerow([i+1,y_test[i]])
