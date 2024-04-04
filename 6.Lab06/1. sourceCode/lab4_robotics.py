from lab2_robotics import * # Includes numpy import

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # Code almost identical to the one from lab2_robotics...

'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint-1]

    '''
        Method that returns the position 2D of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position 2D of the joint
    '''
    def getJointPos2D(self, link_index):
        # get transformation matrix
        T = self.getLinkTranform(link_index)

        return (T[0:2,3]).reshape(2,1)

    '''
        Method that returns the orientation of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): orientation of the joint
    '''
    def getJointOrientation2D(self, link_index): 
        # get transformation matrix
        T = self.getLinkTranform(link_index)
        # Compute yaw_angle
        yaw_angle   = np.arctan2(T[1, 0], T[0, 0])
        return yaw_angle
    

    '''
        Method that returns the configuration of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): orientation of the joint
    '''
    def getJointConfiguration2D(self, link_index): 
        # get transformation matrix
        T = self.getLinkTranform(link_index)

        pos         = (T[0:2,3]).reshape(2,1)
        yaw_angle   = np.arctan2(T[1, 0], T[0, 0])

        return np.block([[pos],
                         [yaw_angle]]).reshape(3,1)
    
    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

    '''
        Method that return the transformation of a selected link

        Argument:
        link_index (integer):

        Returns:
        (np.array((4,4))): transformation matric of the selected link
    '''
    def getLinkTranform(self, link_index: int):
        return self.T[link_index]

    '''
        Method that return the Jacobean for a selected link

        Argument:
        link_index (integer):

        Returns:
        (np.array((6,DoF))): Jacobean matric of the selected link
    '''
    def getLinkJacobean(self, link_index):
        # Transformation matrix from base to each link, until get the selected link
        TT = []
        for i in range(link_index+1):
            TT.append(self.getLinkTranform(i))
        # return Jacobean matrix for selected link
        return jacobian(TT, self.revolute)
    
'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired, robot):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.task_dim = np.shape(desired)[0]                    # Get task dimension
        self.J = np.zeros((self.task_dim, robot.getDOF()))      # Initialize with proper dimensions
        self.err = np.zeros((self.task_dim, 1))                 # Initialize with proper dimensions
        self.link_index = self.name_to_link_index(name, robot.getDOF())      # Get joint number of task from name of the task
        
        self.K = np.eye(self.task_dim,self.task_dim)    # Gain feed forward controller
        self.ffVel = np.eye(self.task_dim,1)    # Feed forward velocity
        self.useFFVel = False                   # Use feed forward velocity

        self.ar = 1                             # Activation function
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method getting link index.

        Arguments:
        name: string
        DoF

        Return:
        link_index: integer
    '''
    def name_to_link_index(self, name: str, DoF):
        if name.split()[0] == "End-effector":
            # End-effector task
            return DoF
        else:
            # Joint i position task
            return int(name.split()[1])
        
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err
    
    '''
        Method returning the gain matrix (K).
    '''    
    def getGainMatrix(self):
        return self.K
    '''
        Method setting the gain matrix (K).
    '''    
    def setGainMatrix(self, K):
        self.K = K * np.eye(self.task_dim,self.task_dim)
        return True
    
    '''
        Method returning the feed-forward velocity.
    '''    
    def getFFVelocity(self):
        return self.ffVel
    '''
        Method setting the gain matrix (K).
    '''    
    def setFFVelocity(self, ffVel):
        self.ffVel = ffVel
        return True
    '''
        Method setting use feed-forward velocity.
    '''    
    def setUseFFVelocity(self, useFFVel):
        self.useFFVel = useFFVel
        return True
    
    '''
        Method checking if task is active
    '''
    def isActive(self) -> int:
        return self.ar

    

'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
  
    def __init__(self, name, desired, robot):
        super().__init__(name, desired, robot)

    def update(self, robot, dt: float, newDesired):
        DoF = robot.getDOF()
        # Update Jacobean matrix - task Jacobian
        self.J  = robot.getLinkJacobean(self.link_index)[[0,1]].reshape(self.task_dim,DoF)  
        # Update task error
        self.err = self.getDesired() - robot.getJointPos2D(self.link_index)
        # Compute feed-forward velocity
        self.ffVel = (newDesired - self.getDesired()) / dt
        # Set new desired
        self.setDesired(newDesired)
        return True
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, robot):
        super().__init__(name, desired, robot)

    def update(self, robot, dt: float, newDesired):
        DoF = robot.getDOF()
        # Update Jacobean matrix - task Jacobian
        self.J  = robot.getLinkJacobean(self.link_index)[[5]].reshape(self.task_dim,DoF)   
        # Update task error
        self.err = self.getDesired() - robot.getJointOrientation2D(self.link_index)
        # Compute feed-forward velocity
        self.ffVel = (newDesired - self.getDesired()) / dt
        # Set new desired
        self.setDesired(newDesired)
        return True
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, robot):
        super().__init__(name, desired, robot)

    def update(self, robot, dt: float, newDesired):
        DoF = robot.getDOF()
        # Update Jacobean matrix - task Jacobian
        self.J  = robot.getLinkJacobean(self.link_index)[[0,1,5]].reshape(self.task_dim,DoF)   
        # Update task error
        self.err = self.getDesired() - robot.getJointConfiguration2D(self.link_index)
        # Compute feed-forward velocity
        self.ffVel = (newDesired - self.getDesired()) / dt
        # Set new desired
        self.setDesired(newDesired)
        return True
'''
    Subclass of Task, representing a two-dimensional circular obstacle avoidance action (inequality task).
'''
class Obstacle2D(Task):
  
    def __init__(self, name, obstacle_pos, obstacle_r, robot):
        super().__init__(name, obstacle_pos, robot)
        self.obstacle_r = obstacle_r
        self.distance_to_obstacle = np.zeros((self.task_dim, 1))                 # Initialize with proper dimensions
        self.ar = 0

    def update(self, robot, dt: float, newDesired):
        DoF = robot.getDOF()
        # Update Jacobean matrix - task Jacobian
        self.J  = robot.getLinkJacobean(self.link_index)[[0,1]].reshape(self.task_dim,DoF)  
        # Update task error
        self.distance_to_obstacle = robot.getJointPos2D(self.link_index) - self.getDesired()
        
        self.err = self.distance_to_obstacle / np.linalg.norm(self.distance_to_obstacle)
        # Compute activation function
        if self.ar == 0 and np.linalg.norm(np.linalg.norm(self.distance_to_obstacle)) <= self.obstacle_r[0]:
            self.ar = 1
        elif self.ar == 1 and np.linalg.norm(np.linalg.norm(self.distance_to_obstacle)) >= self.obstacle_r[1]:
            self.ar = 0

        return True
'''
    Subclass of Task, representing joint limits (inequality task).
'''
class Limit2D(Task):
  
    def __init__(self, name, limit_range, threshold, robot):
        super().__init__(name, limit_range, robot)
        self.threshold = threshold      # Threshold [alpha, sigma].reshape(2,1)
        self.ar = 0                     # Initialise activation function is 0


    def update(self, robot, dt: float, newDesired):
        DoF = robot.getDOF()
        # Update Jacobean matrix - task Jacobian
        self.J  = np.zeros((1,DoF))
        self.J[0,self.link_index-1] = 1
        # Update task error
        q_i = robot.getJointPos(self.link_index)
        
        self.err = np.array([1.0]).reshape(1,1)
        # Compute activation function
        if self.ar == 0 and q_i >= self.getDesired()[0,1] - self.threshold[0]:
            self.ar = -1
        elif self.ar == 0 and q_i <= self.getDesired()[0,0] + self.threshold[0]:
            self.ar = 1
        elif self.ar == -1 and q_i <= self.getDesired()[0,1] - self.threshold[1]:
            self.ar = 0
        elif self.ar == 1 and q_i >= self.getDesired()[0,0] + self.threshold[1]:
            self.ar = 0

        return True

''' 
    Subclass of Task, representing the joint position task.
'''
# class JointPosition(Task):
