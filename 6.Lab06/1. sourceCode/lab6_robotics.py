from lab4_robotics import *

deg90 = np.pi / 2

class MobileManipulator:
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
        self.revoluteExt = [True, False] + revolute  # List of joint types extended with base joints
        self.r = 0.25                        # Distance from robot centre to manipulator base
        self.dof = len(self.revoluteExt) # Number of DOF of the system
        self.q = np.zeros((len(self.revolute),1)) # Vector of joint positions (manipulator)
        self.eta = np.zeros((3,1)) # Vector of base pose (position & orientation)
        self.update(np.zeros((self.dof,1)), 0.0) # Initialise robot state

    '''
        Method that updates the state of the robot.

        Arguments:
        dQ (Numpy array): a column vector of quasi velocities
        dt (double): sampling time
    '''
    def update(self, dQ, dt):
        # Update manipulator
        self.q += dQ[2:, 0].reshape(-1,1) * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]

        # Update mobile base pose
        forward_vel     = dQ[1, 0]
        angular_vel     = dQ[0, 0]
        yaw             = self.eta[2, 0]

        self.eta += dt * np.array([forward_vel * np.cos(yaw), 
                                   forward_vel * np.sin(yaw),
                                   angular_vel]).reshape(3, 1)
        
        # Base kinematics
        x, y, yaw = self.eta.flatten()
        Tb = translation2D(x, y) @ rotation2D(yaw)  # Transformation of the mobile base
        
        ### Additional rotations performed, to align the axis:
        # Rotate Z +90 (using the theta of the first base joint)
        # Rotate X +90 (using the alpha of the first base joint)
        ## Z now aligns with the forward velocity of the base
        # Rotate X -90 (using the alpha of the second base joint)
        ## Z is now back to vertical position
        # Rotate Z -90 (using the theta of the first manipulator joint)
        
        # Modify the theta of the base joint, to account for an additional Z rotation
        self.theta[0] -= deg90

        # Combined system kinematics (DH parameters extended with base DOF)
        thetaExt    = np.concatenate([np.array([deg90,      0]),    self.theta])
        dExt        = np.concatenate([np.array([0,     self.r]),    self.d])
        aExt        = np.concatenate([np.array([0,          0]),    self.a])
        alphaExt    = np.concatenate([np.array([deg90, -deg90]),    self.alpha])

        self.T      = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revoluteExt)

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
        return self.q[joint-3]

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

    def getBasePose(self):
        return self.eta

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
        return jacobian(TT, self.revoluteExt)
    

