import numpy as np
import matplotlib.pyplot as plt
from Lifting_line import BEM

a_ind_wake_lst=np.linspace(0,1,10)
res=20
bem=BEM(2)
i=0
for a_ind_wake in a_ind_wake_lst:
    bem.Lifting_line(20,a_ind_wake)





