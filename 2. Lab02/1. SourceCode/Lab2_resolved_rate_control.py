# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 
revolute = [True, True]
sigma_d = np.array([-1.0, 0.5])
K = np.diag([1, 1])

# Simulation params
dt = 1.0/60.0

# Drawing preparation the visualisation of the robot structure in motion
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Drawing preparation the evolution of robot’s joints positions over time
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, autoscale_on=False)
E, = ax2.plot([], [], 'b-', label='DLS') # 
ax2.set_title('Resolved-rate motion control')
ax2.set_xlabel('Time[s]')
ax2.set_ylabel('Error[m]')
ax2.set_xlim(0,10)
ax2.set_ylim(0,2.4)
ax2.legend()
ax2.grid()
EE = []
time_store = []

# Setup controller type and open .txt file to save data later
# 0. Transpose
# 1. Preudoinverse
# 2. DLS
control_type = 2
f1 = open('timestamp.txt', 'w')
f2 = open('DLS.txt', 'w')

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    E.set_data([], [])
    return line, path, point, E

# Simulation loop
def simulate(t):
    global d, q, a, alpha, revolute, sigma_d
    global PPx, PPy

    # Update robot
    T = kinematics(d, q, a, alpha)
    J = jacobian(T, revolute, a, q)

    # Update control
    P = robotPoints2D(T)
    sigma = np.array([P[0,-1], P[1,-1]])        # Position of the end-effector
    err = sigma_d - sigma                       # Control error

    # Choose controller type
    # 0. Transpose
    # 1. Preudoinverse
    # 2. DLS
    if control_type == 0:
        dq = (J[0:2,0:2].T @ err.T).T
    elif control_type == 1:
        dq = (np.linalg.inv(J[0:2,0:2]) @ err.T).T
    else:
        dq = (DLS(J[0:2,0:2],0.2) @ err.T).T
    
    q += dt * dq
    
    # Store data to plot
    time_store.append(t)
    err_norm = np.linalg.norm(err)
    EE.append(err_norm)
    E.set_data(time_store, EE)
    # Write to .txt file
    f1.write(str(t) + '\n')
    f2.write(str(err_norm) + '\n')

    # Update drawing
    line.set_data(P[0,:], P[1,:])
    PPx.append(P[0,-1])
    PPy.append(P[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])
    if t == 10:
        f1.close()
        f2.close()
    return line, path, point, E

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)

plt.show()