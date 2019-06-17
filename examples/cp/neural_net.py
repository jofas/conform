import numpy as np

from keras.layers import Dense
from keras.models import Sequential

from libconform import CP
from libconform.ncs import NCSNeuralNet

from sklearn.datasets import load_iris

def compile_model(in_dim, out_dim):

    # keras
    model = Sequential()

    model.add(Dense( units=10, activation='tanh'
                   , input_dim=in_dim ))
    model.add(Dense(units=10, activation='tanh'))
    model.add(Dense( units=out_dim
                   , activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

epsilons = [0.01,0.025,0.05,0.1]

X, y = load_iris(True)

labels = np.unique(y)

y = np.array([[0. if j != v else 1.0 for j in labels] for v in y])

# randomly permute X, y
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, y_train = X[:-25], y[:-25]
X_test,  y_test  = X[-25:], y[-25:]

model = compile_model(X.shape[1], y.shape[1])

train = lambda X, y: model.fit(X, y, epochs=5)
predict = lambda X: model.predict(X)

# first training on the majority of data, then the last 25 examples online
ncs = NCSNeuralNet(train, predict)
cp = CP(ncs, epsilons)

cp.train(X_train, y_train)
res = cp.score_online(X_test, y_test)
print(res)
