# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d = np.zeros(3)                 # displacement along Z-axis
q = np.array([0.2, 0.5, 0.2])   # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.3])  # displacement along X-axis
alpha = np.zeros(3)             # rotation around X-axis 
revolute = [True, True, True]   # flags specifying the type of joints
n_DoF = len(revolute)           # Number of Degree of Freedom

# Setting desired position of end-effector to the current one
sigma_d = np.array([1.0, 1.0])

# Simulation params
dt = 1.0/60.0
Tt = 60 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Drawing preparation the evolution of robot’s joints positions over time
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, autoscale_on=False)
q1, = ax2.plot([], [], 'b-', label='q1') # 
q2, = ax2.plot([], [], 'r-', label='q2') # 
q3, = ax2.plot([], [], 'g-', label='q3') # 
ax2.set_title('Joint position')
ax2.set_xlabel('Time[s]')
ax2.set_ylabel('Angel[rad]')
ax2.set_xlim(0,Tt)
ax2.set_ylim(-2,2)
ax2.legend()
ax2.grid()

# Memory
PPx = []
PPy = []
q1_store = []
q2_store = []
q3_store = []
time_store = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    q1.set_data([], [])
    q2.set_data([], [])
    q3.set_data([], [])
    return line, path, point, q1, q2, q3

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma_d
    global PPx, PPy
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)
    
    # Update control
    P           = robotPoints2D(T)
    sigma       = np.array([P[0,-1], P[1,-1]])           # Current position of the end-effector
    err         = sigma_d - sigma                        # Error in position
    Jbar        = J[0:2,0:n_DoF]                         # Task Jacobian
    P           = np.eye(3,3) - DLS(Jbar, 0.0) @ Jbar    # Null space projector
    y           = np.array([-5 + 10 * np.sin(0.5 * t),
                            -5 + 10 * np.sin(0.1 * t),
                            -5 + 10 * np.sin(1.0 * t)]) # Arbitrary joint velocity
    dq          = DLS(Jbar,0.2) @ err + P @ y           # Control signal
    q           = q + dt * dq                           # Simulation update

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])

    time_store.append(t)
    q1_store.append(normalize_angle(q[0]))
    q1.set_data(time_store, q1_store)
    q2_store.append(normalize_angle(q[1]))
    q2.set_data(time_store, q2_store)
    q3_store.append(normalize_angle(q[2]))
    q3.set_data(time_store, q3_store)

    return line, path, point, q1, q2, q3
# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, Tt, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()