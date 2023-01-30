import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)
y = np.average(x)
x_labels = ['Day1', 'Day2', 'Day3', 'Day4']
y_labels = ['Day5']

x = np.ones((16, 16))
x[4:6, 4:6] = 0
x[8:10, 8:10] = 2
x[1:3, 1:3] = 3
x[11:12, 11:12] = 10
plt.imshow(x,  # alpha=lab_alpha,
                 # vmin=np.quantile(ce * weight, 0.1), vmax=np.quantile(ce * weight, 0.9),
                 cmap='jet')
plt.colorbar()
plt.show()

plt.imshow(x,  # alpha=lab_alpha,
                 # vmin=np.quantile(ce * weight, 0.1), vmax=np.quantile(ce * weight, 0.9),
                 cmap='seismic')
plt.colorbar()
plt.show()

plt.scatter(x_labels, x)
plt.scatter(y_labels, y)
plt.show()

x = np.arange(4)
y = np.average(x)

x = np.arange(4)*2 + 1
y = np.average(x)
plt.scatter(x_labels, x)
plt.scatter(y_labels, y)
plt.show()

x = np.arange(4)*2 + 1
y = np.average(x**2)
plt.scatter(x_labels, x)
plt.scatter(y_labels, y)
plt.show()