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
dq_max = np.array([3, 3, 3]) # The maximum joint velocity limit
# Desired values of task variables
sigma1_d = np.array([-1.0 + 2.0 * np.random.rand(), -1.0 + 2.0 * np.random.rand()])     # Position of the end-effector
sigma2_d = np.array([0.0])                                                              # Position of joint 1

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector
N_iter = -1
time = 0

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
q1, = ax2.plot([], [], 'b-', label='e1 (end-effector position)') # 
q2, = ax2.plot([], [], color='orange', label='e2 (joint 1 position)') # 
# q3, = ax2.plot([], [], 'g-', label='q3') # 
ax2.set_title('Joint position')
ax2.set_xlabel('Time[s]')
ax2.set_ylabel('Error')
ax2.set_xlim(0,10*Tt)
ax2.set_ylim(-0.1,2.5)
ax2.set_xticks(range(0, 10*Tt, 10))
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
    # q3.set_data([], [])
    return line, path, point, q1, q2#, q3

# Simulation loop
def simulate(t):
    
    global q, a, d, alpha, revolute, dq_max, sigma1_d, sigma2_d
    global PPx, PPy
    global time, N_iter, Tt
    
    # Set new desired end-effector position and joint 1 position at beginning of each 10s
    if t == 0:
        # Set random new desired position
        theta_rand = 2 * math.pi * np.random.rand()
        length_rand = sum(a[1:3]) * np.random.rand()
        sigma1_d = np.array([length_rand * np.cos(theta_rand) + a[0], length_rand * np.sin(theta_rand)])       # Position of the end-effector
        sigma2_d = np.array([0.0])                                                                      # Position of joint 1
        # Set number of iteration
        N_iter += 1
        

    # Update robot
    T       = kinematics(d, q.flatten(), a, alpha)
    J       = jacobian(T, revolute)
    Probot  = robotPoints2D(T)

    # Update control
    # End-effector position task at the top of the hierarchy
    # TASK 1: Position of End-Effector
    sigma1      = np.array([Probot[0,-1], Probot[1,-1]])                # Current position of the end-effector
    err1        = sigma1_d - sigma1                                     # Error in Cartesian position
    J1          = J[0:2,0:n_DoF]                                        # Jacobian of the first task
    P1          = np.eye(3,3) - DLS(J1, 0.0) @ J1                       # Null space projector
    
    # TASK 2: Position of Joint 1
    sigma2      = np.array([Probot[1,1]])                               # Current position of joint 1
    err2        = sigma2_d - sigma2                                     # Error in joint position
    J2          = jacobian([T[0], T[1]], revolute)[1:2]                 # Jacobian of the second task
    J2bar       = J2 @ P1                                               # Augmented Jacobian
    
    # Combining tasks
    dq1         = DLS(J1,0.1) @ err1                                    # Velocity for the first task
    dq12        = dq1 + DLS(J2bar, 0.2) @ (err2 - J2 @ dq1)             # Velocity for both tasks
    limited_dq12 = np.where(np.abs(dq12) > dq_max, np.sign(dq12) * dq_max, dq12)

    q = q + limited_dq12 * dt                                           # Simulation update

    # Joint position task at the top of the hierarchy
    # # TASK 1: Position of Joint 1
    # sigma2      = np.array([Probot[1,1]])                               # Current position of joint 1
    # err2        = sigma2_d - sigma2                                     # Error in joint position
    # J2          = jacobian([T[0], T[1]], revolute)[1:2]                 # Jacobian of the second task
    # J2bar       = J2                                                    # Augmented Jacobian
    # P2          = np.eye(3,3) - DLS(J2bar, 0.0) @ J2bar                 # Null space projector

    # # TASK 2: Position of End-Effector
    # sigma1      = np.array([Probot[0,-1], Probot[1,-1]])                # Current position of the end-effector
    # err1        = sigma1_d - sigma1                                     # Error in Cartesian position
    # J1          = J[0:2,0:n_DoF]                                        # Jacobian of the first task
    # J1bar       = J1 @ P2                                               # Augmented Jacobian
    
    # # Combining tasks
    # dq2         = DLS(J2,0.1) @ err2                                    # Velocity for the first task
    # dq21        = dq2 + DLS(J1bar, 0.2) @ (err1 - J1 @ dq2)             # Velocity for both tasks
    # limited_dq21 = np.where(np.abs(dq21) > dq_max, np.sign(dq21) * dq_max, dq21)

    # q = q + limited_dq21 * dt                                           # Simulation update

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])

    time_store.append(t + N_iter * Tt)
    q1_store.append(np.linalg.norm(err1))
    q1.set_data(time_store, q1_store)
    q2_store.append(np.linalg.norm(err2))
    q2.set_data(time_store, q2_store)

    return line, path, point, q1, q2

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, Tt, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)

plt.show()