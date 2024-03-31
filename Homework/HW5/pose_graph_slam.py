import numpy as np
import gtsam
from typing import NamedTuple

def pretty_print(arr):
    return '\n'.join([' '.join(['%.2f' % x for x in c]) for c in arr])

class Pose2(NamedTuple):
    '''
    Pose2 class for 2D pose
    @usage: pose = Pose2(id, x, y, z)
            print(pose.x)
    '''
    id: int
    x: float
    y: float
    theta: float

class Edge2(NamedTuple):
    '''
    Edge2 class for 2D edge
    @usage: edge = Edge2(id1, id2, x, y, z, info)
            print(edge.x)
    '''
    id1: int
    id2: int
    x: float
    y: float
    theta: float
    info: np.ndarray # 3x3 matrix

    def __str__(self):
        return f"Edge2(id1={self.id1}, id2={self.id2}, x={self.x}, y={self.y}, theta={self.theta},\ninfo=\n{pretty_print(self.info)})\n"

class Pose3(NamedTuple):
    '''
    Pose3 class for 3D pose
    @usage: pose = Pose3(id, x, y, z, qx, qy, qz, qw)
            print(pose.x)
    '''
    id: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

class Edge3(NamedTuple):
    '''
    Edge3 class for 3D edge
    @usage: edge = Edge3(id1, id2, x, y, z, qx, qy, qz, qw, info)
            print(edge.x)
    '''
    id1: int
    id2: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    info: np.ndarray # 6x6 matrix

    def __str__(self):
        return f"Edge3(id1={self.id1}, id2={self.id2}, x={self.x}, y={self.y}, z={self.z}, qx={self.qx}, qy={self.qy}, qz={self.qz}, qw={self.qw},\ninfo=\n{pretty_print(self.info)})\n"


def read_g2o_2d(file_name):
    data = {
        'poses': [],
        'edges': []
    }

    # read the file
    with open(file_name, "r") as f:
        lines = f.readlines()

        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        # fill in the `data` dict with Pose2 or Edge2 objects
        
        # ...

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    return data

def gn_2d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create an empty factor graph
    # graph = ...
    # initial_values = ...

    # set initial_values according to poses
    # ...

    # add prior factor for the first pose
    # ...

    # add between factors according to edges
    # ...

    # optimize the graph
    # ...
    # result = ...

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose2(result)

def isam_2d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    # ...

    for pose in poses:

        frame_id = pose.id

        # create an empty factor graph
        # graph = ...
        # initial_values = ...
        
        if frame_id==0:
            # initialization
            # ...
            pass
        else:
            # optimize new frame
            # ...
            pass

        # update isam
        # ...
        # result = ...

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose2(result)

def read_g2o_3d(file_name):
    data = {
        'poses': [],
        'edges': []
    }


    # read the file
    with open(file_name, "r") as f:
        lines = f.readlines()

        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        # fill in the `data` dict with Pose3 or Edge3 objects

        # ...

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    
    return data  

def gn_3d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()

    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create an empty factor graph
    # graph = ...
    # initial_values = ...

    # set initial_values according to poses
    # ...

    # add prior factor for the first pose
    # ...

    # add between factors according to edges
    # ...

    # optimize the graph
    # ...
    # result = ...

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose3(result)

def isam_3d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()

    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    # ...

    for pose in poses:

        frame_id = pose.id

        # create an empty factor graph
        # graph = ...
        # initial_values = ...
        
        if frame_id==0:
            # initialization

            # Notice that the order of quaternion is different from
            # the one in the g2o file. GTSAM uses (qw, qx, qy, qz).

            # ...
            pass
        else:
            # optimize new frame
            # ...
            pass

        # update isam
        # ...
        # result = ...

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose3(result)
