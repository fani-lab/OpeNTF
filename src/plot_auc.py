import matplotlib.pyplot as plt
import numpy as np

# Fnn
x_fnn = np.array([25, 50, 100])

# lr = 0.01
# y_fnn = np.array([0.760248447204968, 0.769565217391304, 0.786645962732919])
# var_fnn = np.array([0.00158230778133559, 0.000465452721731413, 0.00124185023725936])

# lr = 0.03
y_fnn = np.array([0.775155279503105, 0.793788819875776, 0.761490683229813])
var_fnn = np.array([0.00156186103931175, 0.00101404266810693, 0.00295571158520119])

plt.plot(x_fnn, y_fnn, 'k-')
plt.fill_between(x_fnn, y_fnn - var_fnn,  y_fnn + var_fnn, label='fnn')

# Sgns
x_sgns = np.array([25, 50, 100])

# lr = 0.01
# y_sgns = np.array([0.818012422360248, 0.795341614906832, 0.788819875776397, ])
# var_sgns = np.array([0.00102272288877743, 0.00228907063770687, 0.000709849157054123])

# lr = 0.03
y_sgns = np.array([0.751863354037267, 0.770496894409937, 0.768322981366459])
var_sgns = np.array([0.0024240962925813, 0.00541395007908645, 0.00237008603063153])

plt.plot(x_sgns, y_sgns, 'k-')
plt.fill_between(x_sgns, y_sgns - var_sgns, y_sgns + var_sgns, label='sgns')

plt.xlabel('Number of nodes')
plt.ylabel('AUC')
plt.title('lr = 0.03')
plt.legend()
plt.show()
