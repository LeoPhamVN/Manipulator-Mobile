from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model - 3-link manipulator
d       = np.zeros(3)                   # displacement along Z-axis
theta   = np.array([0.2, 0.5, 0.2])     # rotation around Z-axis (theta)
a       = np.array([0.4, 0.3, 0.2])    # displacement along X-axis
alpha   = np.zeros(3)                   # rotation around X-axis 
revolute = [True, True, True]           # flags specifying the type of joints
robot   = MobileManipulator(d, theta, a, alpha, revolute) # Manipulator object
n_DoF   = robot.getDOF()

# Task definition

tasks = [ 
          Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), robot)
        ] 
# Controller params
Weight = np.diag([2, 2, 0.2, 0.2, 0.2])
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
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    global tasks, N_iter
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    # Set random new desired position
    theta_rand = 2 * math.pi * np.random.rand()
    length_rand = np.random.uniform(1.0, sum(a[0:3]))
    tasks = [ 
          Position2D("End-effector position", np.array([length_rand * np.cos(theta_rand), length_rand * np.sin(theta_rand)]).reshape(2,1), robot)
    ] 

    # Set number of iteration
    N_iter += 1
    # Set gain matrix
    tasks[-1].setGainMatrix(1.0)
    tasks[-1].setUseFFVelocity(False)

    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global time, N_iter, Tt
    
    # Set desired trajectory
    newDesired = [tasks[0].getDesired()]

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
            dq      = dq + DLS_Weighted(Jbar, 0.2, Weight) @ (int(tasks[i].useFFVel) * tasks[i].ffVel + tasks[i].isActive() * tasks[i].K @ tasks[i].err - tasks[i].J @ dq) 
            # Update null-space projector
            P       = P - DLS_Weighted(Jbar, 0.001, Weight) @ Jbar  
        else:
            dq      = dq
            P       = P 
    ###

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])
    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)

    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, Tt, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()