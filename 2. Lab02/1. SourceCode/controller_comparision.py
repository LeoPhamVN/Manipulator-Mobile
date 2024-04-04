from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt

# Load the list from the file
timestamp = []
EE1 = []
EE2 = []
EE3 = []
with open('timestamp.txt', 'r') as f:
    for line in f:
        timestamp.append(float(line.strip()))

f.close()

with open('Transpose.txt', 'r') as f:
    for line in f:
        EE1.append(float(line.strip()))

f.close()

with open('Preudoinverse.txt', 'r') as f:
    for line in f:
        EE2.append(float(line.strip()))

f.close()

with open('DLS.txt', 'r') as f:
    for line in f:
        EE3.append(float(line.strip()))

f.close()


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, autoscale_on=False)
E1, = ax3.plot(timestamp, EE1, 'b-', label='Transpose') # 
E2, = ax3.plot(timestamp, EE2, 'g-', label='Preudoinverse') # 
E3, = ax3.plot(timestamp, EE3, 'r-', label='DLS') # 
ax3.set_title('Resolved-rate motion control')
ax3.set_xlabel('Time[s]')
ax3.set_ylabel('Error[m]')
ax3.set_xlim(0,10)
ax3.set_ylim(0,2.4)
ax3.legend()
ax3.grid()

plt.show()
