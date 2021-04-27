import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

house_data = pd.read_csv('house.csv')
house_data=house_data[house_data['loyer']<10000]

# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values]).T
y = np.matrix(house_data['loyer']).T

# On effectue le calcul exact du paramètre theta
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)


plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')

plt.show()


print(theta)