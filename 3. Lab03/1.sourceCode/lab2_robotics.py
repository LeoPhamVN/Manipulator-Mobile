import numpy as np # Import Numpy
import math

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
            d (double)      : displacement along Z-axis
            theta (double)  : rotation around Z-axis
            a (double)      : displacement along X-axis
            alpha (double)  : rotation around X-axis

        Returns:
            (Numpy array)   : composition of elementary DH transformations
    '''
    
    # Calculate trigonometric values
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # 1. Build matrices representing elementary transformations (based on input parameters).
    T1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])

    T2 = np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta,  0, 0],
        [0,         0,          1, 0],
        [0,         0,          0, 1]
    ])

    T3 = np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T4 = np.array([
        [1, 0,          0,          0],
        [0, cos_alpha, -sin_alpha,  0],
        [0, sin_alpha,  cos_alpha,  0],
        [0, 0,          0,          1]
    ])

    # 2. Multiply matrices in the correct order (result in T).
    T = T1@T2@T3@T4

    return T

def kinematics(d, theta, a, alpha):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
            d (list of double)      : list of displacements along Z-axis
            theta (list of double)  : list of rotations around Z-axis
            a (list of double)      : list of displacements along X-axis
            alpha (list of double)  : list of rotations around X-axis

        Returns:
            (list of Numpy array)   : list of transformations along the kinematic chain (from the base frame)
    '''
    T = [np.eye(4)] # Base transformation
    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.
    N_order = len(d)
    for index in range(N_order):
        T.append(T[index] @ DH(d[index], theta[index], a[index], alpha[index]))

    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
            T (list of Numpy array) : list of transformations along the kinematic chain of the robot (from the base frame)
            revolute (list of Bool) : list of flags specifying if the corresponding joint is a revolute joint

        Returns:
            (Numpy array)           : end-effector Jacobian
    '''
    # 1. Initialize J and O.
    J = np.zeros((6,len(revolute)))
    z_pre = T[0][0:3,2].reshape(1,3)
    o_pre = T[0][0:3,3].reshape(1,3)
    
    o_n = T[-1][0:3,3].reshape(1,3)

    # 2. For each joint of the robot
    for index in range(1,len(T)):
        #   a. Extract z and o.
        z = T[index][0:3,2].reshape(1,3)
        o = T[index][0:3,3].reshape(1,3)
        #   b. Check joint type.
        #   c. Modify corresponding column of J.
        J[:,index-1] = np.block([int(revolute[index-1])*np.cross(z_pre, o_n - o_pre) + (1 - int(revolute[index-1]))*z_pre, int(revolute[index-1])*z_pre])
        #   d. Set z and o for next joint
        z_pre = z
        o_pre = o

    return J

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    return A.T @ np.linalg.inv(A @ A.T + damping**2)

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
            T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
            (Numpy array): an array of 2D points
    '''
    # Init P
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        # Get P from transformation matrix
        P[:,i] = T[i][0:2,3]
    return P

def normalize_angle(angle):
    """
    Normalize an angle to be within the range of -pi to pi.
    
    Args:
    angle (float): The angle to be normalized, in radians.
    
    Returns:
    float: The normalized angle.
    """
    while angle < -math.pi:
        angle += 2*math.pi
    while angle > math.pi:
        angle -= 2*math.pi
    return angle