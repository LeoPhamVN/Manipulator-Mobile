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
obstacle_pos1   = np.array([0.0, 1.0]).reshape(2,1)
obstacle_r1     = 0.5
obstacle_pos2   = np.array([-0.5, -0.75]).reshape(2,1)
obstacle_r2     = 0.4
obstacle_pos3   = np.array([0.8, -0.5]).reshape(2,1)
obstacle_r3     = 0.3

tasks = [ 
            Obstacle2D("End-effector Obstacle avoidance", obstacle_pos1, np.array([obstacle_r1, obstacle_r1+0.01]), robot),
            Obstacle2D("End-effector Obstacle avoidance", obstacle_pos2, np.array([obstacle_r2, obstacle_r2+0.01]), robot),
            Obstacle2D("End-effector Obstacle avoidance", obstacle_pos3, np.array([obstacle_r3, obstacle_r3+0.01]), robot),
            Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), robot)
        ] 
n_obstacles = 0
n_positions = 0
for i in range(len(tasks)):
    if type(tasks[i]) == Obstacle2D:
        n_obstacles += 1
    elif type(tasks[i]) == Position2D:
        n_positions += 1
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
ax.add_patch(patch.Circle(obstacle_pos1.flatten(), obstacle_r1, color='red', alpha=0.3))
ax.add_patch(patch.Circle(obstacle_pos2.flatten(), obstacle_r2, color='green', alpha=0.3))
ax.add_patch(patch.Circle(obstacle_pos3.flatten(), obstacle_r3, color='purple', alpha=0.3))
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target

q = [[] ,[], [], []]
# Drawing preparation the evolution of robot’s joints positions over time
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, autoscale_on=False)
q[0], = ax2.plot([], [], 'r-', label='d1 (distance to obstacle)') # 
q[1], = ax2.plot([], [], 'g-', label='d2 (distance to obstacle)') # 
q[2], = ax2.plot([], [], 'y-', label='d3 (distance to obstacle)') # 
q[3], = ax2.plot([], [], 'b-', label='e1 (end-effector position error)') # 
ax2.set_title('Task-Priority inequality tasks')
ax2.set_xlabel('Time[s]')
ax2.set_ylabel('Error')
# ax2.set_xlim(0,10)
# ax2.set_ylim(-0.1,0.8)
# ax2.set_xticks(range(0, 10, 2))
x_length = 6
ax2.set_xlim(0,x_length*Tt)
ax2.set_ylim(-0.1,3.0)
ax2.set_xticks(range(0, x_length*Tt, 10))
ax2.legend()
ax2.grid()

# Memory
PPx = []
PPy = []
q_store = [[], [], [], []]
q_store[0] = []
q_store[1] = []
q_store[2] = []
q_store[3] = []

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
    q[0].set_data([], [])
    q[1].set_data([], [])
    q[2].set_data([], [])
    q[3].set_data([], [])

    # Set random new desired position
    theta_rand = 2 * math.pi * np.random.rand()
    length_rand = np.random.uniform(1.0, sum(a[0:3]))
    tasks = [ 
        Obstacle2D("End-effector Obstacle avoidance", obstacle_pos1, np.array([obstacle_r1, obstacle_r1+0.01]), robot),
        Obstacle2D("End-effector Obstacle avoidance", obstacle_pos2, np.array([obstacle_r2, obstacle_r2+0.01]), robot),
        Obstacle2D("End-effector Obstacle avoidance", obstacle_pos3, np.array([obstacle_r3, obstacle_r3+0.01]), robot),
        Position2D("End-effector position", np.array([length_rand * np.cos(theta_rand), length_rand * np.sin(theta_rand)]).reshape(2,1), robot),
    ] 

    # Set number of iteration
    N_iter += 1
    # Set gain matrix
    tasks[-1].setGainMatrix(1.0)
    tasks[-1].setUseFFVelocity(False)

    return line, path, point, q[0], q[1], q[2], q[3]

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global time, N_iter, Tt
    
    # Set desired trajectory
    newDesired = [tasks[0].getDesired(), tasks[1].getDesired(), tasks[2].getDesired(), tasks[3].getDesired()]

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
    if type(tasks[1]) == Obstacle2D:
        print(1)

    q_store[0].append(np.linalg.norm(tasks[0].distance_to_obstacle)-tasks[0].obstacle_r[0])
    q[0].set_data(time_store, q_store[0])
    
    q_store[1].append(np.linalg.norm(tasks[1].distance_to_obstacle)-tasks[1].obstacle_r[0])
    q[1].set_data(time_store, q_store[1])
    q_store[2].append(np.linalg.norm(tasks[2].distance_to_obstacle)-tasks[2].obstacle_r[0])
    q[2].set_data(time_store, q_store[2])
    q_store[3].append(np.linalg.norm(tasks[-1].err))
    q[3].set_data(time_store, q_store[3])
    
    # q2_store.append(np.linalg.norm(tasks[1].distance_to_obstacle)-tasks[1].obstacle_r[0])
    # q2.set_data(time_store, q2_store)
    # q3_store.append(np.linalg.norm(tasks[2].distance_to_obstacle)-tasks[2].obstacle_r[0])
    # q3.set_data(time_store, q3_store)
    # q4_store.append(np.linalg.norm(tasks[-1].err))
    # q4.set_data(time_store, q4_store)

    return line, path, point, q[0], q[1], q[2], q[3]

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()