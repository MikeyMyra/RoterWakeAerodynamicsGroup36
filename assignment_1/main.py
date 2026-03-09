import matplotlib.pyplot as plt

from question_d import case_d
from question_e import case_e
from question_f import case_f


if __name__ == "__main__":
    
    J_baseline = 2.1428570754 # rpm = 1200
    
    case_d(J_baseline=J_baseline)
    case_e(J_baseline=J_baseline)
    case_f(J_baseline=J_baseline)
    
    plt.show()