from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch

# Robot model - 3-link manipulator
d       = np.zeros(3)                   # displacement along Z-axis
theta   = np.array([0.2, 0.5, 0.2])     # rotation around Z-axis (theta)
a       = np.array([0.75, 0.5, 0.3])    # displacement along X-axis
alpha   = np.zeros(3)                   # rotation around X-axis 
revolute = [True, True, True]           # flags specifying the type of joints
robot   = Manipulator(d, theta, a, alpha, revolute) # Manipulator object
n_DoF   = robot.getDOF()

# Task hierarchy definition
limit_range   = np.array([-0.5, 0.5]).reshape(1,2)
threshold     = np.array([0.05, 0.1]).reshape(2,1)


tasks = [ 
            Limit2D("Joint 1 Limitation", limit_range, threshold, robot),
            Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), robot)
        ] 

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector
N_iter = -3
time = 0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target

# Drawing preparation the evolution of robot’s joints positions over time
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, autoscale_on=False)
q1, = ax2.plot([], [], 'b-', label='q1 (position of joint 1)') # 
q2, = ax2.plot([], [], 'r--') # 
q3, = ax2.plot([], [], 'r--') # 
q4, = ax2.plot([], [], 'g-', label='e1 (end-effector position error)') # 
ax2.set_title('Task-Priority control')
ax2.set_xlabel('Time[s]')
ax2.set_ylabel('Error')
x_length = 6
ax2.set_xlim(0,x_length*Tt)
ax2.set_ylim(-0.1,3.0)
ax2.set_ylim(-0.6,2.0)
ax2.set_xticks(range(0, x_length*Tt, 10))
ax2.legend()
ax2.grid()

# Memory
PPx = []
PPy = []
q1_store = []
q2_store = []
q3_store = []
q4_store = []
time_store = []

# Simulation initialization
def init():
    global tasks, N_iter
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    q1.set_data([], [])
    q2.set_data([], [])
    q3.set_data([], [])
    q4.set_data([], [])

    # Set random new desired position
    theta_rand = 2 * math.pi * np.random.rand()
    length_rand = np.random.uniform(1.0, sum(a[0:3]))
    tasks = [ 
        Limit2D("Joint 1 Limitation", limit_range, threshold, robot),
        Position2D("End-effector position", np.array([length_rand * np.cos(theta_rand), length_rand * np.sin(theta_rand)]).reshape(2,1), robot),
    ] 

    # Set number of iteration
    N_iter += 1
    # Set gain matrix
    tasks[-1].setGainMatrix(1.0)
    tasks[-1].setUseFFVelocity(False)

    return line, path, point, q1, q2, q3, q4

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global time, N_iter, Tt
    
    # Set desired trajectory
    newDesired = [tasks[0].getDesired(), tasks[1].getDesired()]

    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    # Initialize null-space projector
    P   = np.eye(n_DoF, n_DoF)
    # Initialize output vector (joint velocity)
    dq  = np.zeros((n_DoF, 1))
    # Loop over tasks
    for i in range(len(tasks)):      
        # Update task state
        tasks[i].update(robot, dt, newDesired[i])
        if tasks[i].ar != 0:
            # Compute augmented Jacobian
            Jbar    = tasks[i].J @ P 
            # Compute task velocity
            # Accumulate velocity
            dq      = dq + DLS(Jbar, 0.2) @ (int(tasks[i].useFFVel) * tasks[i].ffVel + tasks[i].isActive() * tasks[i].K @ tasks[i].err - tasks[i].J @ dq) 
            # Update null-space projector
            P       = P - DLS(Jbar, 0.001) @ Jbar  
        else:
            dq      = dq
            P       = P 
    ###

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])
    
    time_store.append(t + N_iter * Tt)
    err = []
    for i in range(len(tasks)):
        err.append(np.linalg.norm(tasks[i].err))

    q1_store.append(robot.getJointPos(1).astype(np.float32))
    q1.set_data(time_store, q1_store)
    q2_store.append(tasks[0].getDesired()[0,0])
    q2.set_data(time_store, q2_store)
    q3_store.append(tasks[0].getDesired()[0,1])
    q3.set_data(time_store, q3_store)
    q4_store.append(np.linalg.norm(tasks[-1].err))
    q4.set_data(time_store, q4_store)

    return line, path, point, q1, q2, q3, q4

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()