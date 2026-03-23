import numpy as np
import matplotlib.pyplot as plt

r_norm = np.linspace(0, 1, 50)

chord = 0.18 - 0.06 * r_norm
twist = -50 * r_norm + 46 + 0.7 * 50

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(r_norm, chord, 'b-', marker='')
plt.title("Chord Distribution")
plt.xlabel("r/R")
plt.ylabel("Chord Length (m)")
plt.grid(True)
plt.ylim(0, 0.2)

plt.subplot(1, 2, 2)
plt.plot(r_norm, twist, 'r-', marker='')
plt.title("Twist Distribution")
plt.xlabel("r/R")
plt.ylabel("Twist Angle (deg)")
plt.grid(True)
plt.ylim(0, 90)

plt.tight_layout()
plt.show()