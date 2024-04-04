# Import necessary libraries
from lab2_robotics import * # Import our library (includes Numpy)
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (planar 2 link manipulator)
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 

# Simulation params
dt = 0.1 # Sampling time
Tt = 80 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation the visualisation of the robot structure in motion
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
line, = ax1.plot([], [], 'o-', lw=2) # Robot structure
path, = ax1.plot([], [], 'r-', lw=1) # End-effector path
ax1.set_title('Kinematics')
ax1.set_xlabel('x[m]')
ax1.set_ylabel('y[m]')
ax1.set_aspect('equal')
ax1.grid()


# Drawing preparation the evolution of robot’s joints positions over time
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, autoscale_on=False)
q1, = ax2.plot([], [], 'b-', label='q1') # 
q2, = ax2.plot([], [], 'r-', label='q2') # 
ax2.set_title('Joint position')
ax2.set_xlabel('Time[s]')
ax2.set_ylabel('Angel[rad]')
ax2.set_xlim(0,Tt)
ax2.set_ylim(0,30)
ax2.legend()
ax2.grid()

# Memory
PPx = []
PPy = []
q1_store = []
q2_store = []
time_store = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    q1.set_data([], [])
    q2.set_data([], [])
    return line, path, q1, q2

# Simulation loop
def simulate(t):
    global d, q, a, alpha
    global PPx, PPy
    
    # Update robot
    T = kinematics(d, q, a, alpha)
    dq = np.array([0.1, 0.3]) # Define how joint velocity changes with time!
    q[0] += dt * dq[0]
    q[1] += dt * dq[1]
    
    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    
    time_store.append(t)
    q1_store.append(q[0])
    q1.set_data(time_store, q1_store)
    q2_store.append(q[1])
    q2.set_data(time_store, q2_store)
    return line, path, q1, q2

# Run simulation
animation = anim.FuncAnimation(fig1, simulate, tt, 
                                interval=10, blit=True, init_func=init, repeat=False)



plt.show()