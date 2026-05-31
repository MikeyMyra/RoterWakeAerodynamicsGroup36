import numpy as np
import matplotlib.pyplot as plt

data_lin=np.loadtxt('assignment_2\\Sensitivity_analysis\\Finest_grid.txt')
N_lin=len(data_lin[0,:])
# print(data_lin[0,:])
data_cos=np.loadtxt('assignment_2\\Sensitivity_analysis\\Finest_grid_cosine.txt')
N_cos=len(data_cos[0,:])


plt.plot(data_lin[0,:],data_lin[1,:],label=f'res={N_lin}')
plt.plot(data_cos[0,:],data_cos[1,:],label=f'res={N_cos}')
plt.legend()
plt.grid()
plt.show()



