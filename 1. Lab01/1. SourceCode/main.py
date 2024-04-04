# Library import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Simulation params
T = 5*2*np.pi # Simulation time
dt = 1.0/60.0 # Sampling time

# Drawing preparation (figure with one subplot)
fig = plt.figure() 
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set(xlabel='x[m]', ylabel='y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Line for displaying the structure
path, = ax.plot([], [], 'r-', lw=1) # Line for displaying the path

# Path memory
Px = []; Py = []

# Function initializing simulation and drawing
def init():
    line.set_data([], [])
    path.set_data([], [])
    return line, path

# Function updating the simulation (main loop)
def simulate(t):
    # Computing new position based on time
    P = np.zeros((2,2))
    r = 1.0 + 0.2 * np.sin(t*7.2)
    P[0,1] = np.cos(t) * r
    P[1,1] = np.sin(t) * r
    
    # Saving new point in path memory
    Px.append(P[0,1])
    Py.append(P[1,1])
    
    # Updating the drawing
    line.set_data(P[0,:], P[1,:])
    path.set_data(Px, Py)
    return line, path

# Create and run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, T, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()
#animation.save('anim.mp4', fps=60) # To save simulation to a file