import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(1)
import keras.optimizers
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

np.random.seed(seed=1)
N = 200
K = 3
T = np.zeros((N, 3), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3]
X_range1 = [-3, 3]
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])
Pi = np.array([0.4, 0.8, 1])
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T[n, k] = 1
            break

    for k in range(2):
        X[n, k] = np.random.randn() * Sig[T[n, :] == 1, k] + Mu[T[n, :] == 1, k]    

TrainingRatio = 0.5
X_n_training = int(N * TrainingRatio)
X_train = X[:X_n_training, :]        
X_test = X[X_n_training:, :] 
T_train = T[:X_n_training, :]  
T_test = T[X_n_training:, :] 

np.savez("class_data.npz", X_train=X_train, T_train=T_train, X_test=X_test, T_test=T_test, X_range0=X_range0, X_range1=X_range1)

outfile = np.load("class_data.npz")
X_train = outfile["X_train"]
T_train = outfile["T_train"]
X_test = outfile["X_test"]
T_test = outfile["T_test"]
X_range0 = outfile["X_range0"]
X_range1 = outfile["X_range1"]

def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], linestyle="none", marker="o", markeredgecolor="black", color=c[i], alpha=0.8)
    plt.grid(True)  

np.random.seed(1)

model = Sequential()
model.add(Dense(2, input_dim=2, activation="sigmoid", kernel_initializer='uniform'))
model.add(Dense(3, input_dim=2, activation="softmax", kernel_initializer='uniform'))
sgd = keras.optimizers.SGD(lr=0.5, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

startTime = time.time()
history = model.fit(X_train, T_train, epochs=1000, batch_size=100, verbose=0, validation_data=(X_test, T_test))

score = model.evaluate(X_test, T_test, verbose=0)
print("cross entropy{0:.2f}, accuracy{1:.2f}".format(score[0], score[1]))
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f}sec".format(calculation_time))

plt.figure(1, figsize=(12, 3))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1, 3, 1)
plt.plot(history.history["loss"], "black", label="training")
plt.plot(history.history["val_loss"], "cornflowerblue", label="test")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history["accuracy"], "black", label="training")
plt.plot(history.history["val_accuracy"], "cornflowerblue", label="test")
plt.legend()

plt.subplot(1, 3, 3)
Show_data(X_test, T_test)
xn = 60
x0 = np.linspace(X_range0[0], X_range0[1], xn)
x1 = np.linspace(X_range1[0], X_range1[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
x = np.c_[np.reshape(xx0, xn*xn, 1), np.reshape(xx1, xn*xn, 1)]
y = model.predict(x)
K = 3

for ic in range(K):
    f = y[:, ic]
    f = f.reshape(xn, xn)
    f = f.T
    cont = plt.contour(xx0, xx1, f, levels=[0.5, 0.9], colors=["cornflowerblue", "black"])
    cont.clabel(fmt="%.1f", fontsize=9)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
  