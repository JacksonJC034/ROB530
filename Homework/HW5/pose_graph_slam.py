import numpy as np
import gtsam
from typing import NamedTuple
import os

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
        for line in lines:
            parts = line.strip().split()
            
            if parts[0] == "VERTEX_SE2":
                # Parsing the pose information
                pose_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                theta = float(parts[4])
                data['poses'].append(Pose2(pose_id, x, y, theta))
                
            elif parts[0] == "EDGE_SE2":
                # Parsing the edge information
                from_id = int(parts[1])
                to_id = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                theta = float(parts[5])
                info_array = np.array(parts[6:], dtype=float)
                
                # Construct the information matrix from its upper-triangular form
                info_matrix = np.zeros((3, 3))
                info_matrix[0, 0] = info_array[0]
                info_matrix[0, 1] = info_matrix[1, 0] = info_array[1]
                info_matrix[0, 2] = info_matrix[2, 0] = info_array[2]
                info_matrix[1, 1] = info_array[3]
                info_matrix[1, 2] = info_matrix[2, 1] = info_array[4]
                info_matrix[2, 2] = info_array[5]
                
                data['edges'].append(Edge2(from_id, to_id, x, y, theta, info_matrix))
        # ...

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    return data

def gn_2d(data):
    poses = data['poses']
    edges = data['edges']
    
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # create an empty factor graph
    graph = gtsam.NonlinearFactorGraph()
    result = gtsam.Values()
    # Use a diagonal noise model for the prior factor of the first pose
    prior_noise = gtsam.noiseModel.Diagonal.Variances(first_pose_prior_cov)
    
    # Create a dictionary for initial values
    initial = gtsam.Values()

    # set initial_values according to poses
    for pose in poses:
        initial.insert(pose.id, gtsam.Pose2(pose.x, pose.y, pose.theta))

    # add prior factor for the first pose
    first_pose = poses[0]
    graph.add(gtsam.PriorFactorPose2(first_pose.id, initial.atPose2(0), prior_noise))

    # add between factors according to edges
    for edge in edges:
        # Convert the information matrix to a covariance matrix and then to a noise model
        info = np.linalg.inv(edge.info)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(info)
        graph.add(gtsam.BetweenFactorPose2(edge.id1, edge.id2, gtsam.Pose2(edge.x, edge.y, edge.theta), noise_model))

    # optimize the graph
    params = gtsam.GaussNewtonParams()
    # Set maximum iterations if needed: params.setMaxIterations(1000)
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
    result = optimizer.optimize()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose2(result)

def isam_2d(data):
    poses = data['poses']
    edges = data['edges']
    
    # ISAM2 parameters and initialization
    parameters = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(parameters)
    
    # Create an empty dictionary for initial values
    
    result = gtsam.Values()
    
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    # ...

    for pose in poses:
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        
        if pose.id == 0:
            initialPose = gtsam.Pose2(pose.x, pose.y, pose.theta)
            initial.insert(pose.id, initialPose)
            prior_noise = gtsam.noiseModel.Diagonal.Variances(first_pose_prior_cov)
            graph.add(gtsam.PriorFactorPose2(pose.id, initialPose, prior_noise))
        else:
            #optimize new frame 
            prev_pose = result.atPose2(pose.id - 1)
            initial.insert(pose.id, prev_pose)
            for edge in edges:
                if edge.id2 == pose.id:
                    info = np.linalg.inv(edge.info)
                    noise_model = gtsam.noiseModel.Gaussian.Covariance(info)
                    graph.add(gtsam.BetweenFactorPose2(edge.id1, edge.id2, gtsam.Pose2(edge.x, edge.y, edge.theta), noise_model))
        
        # update isam
        isam.update(graph, initial)
        result = isam.calculateEstimate()
        


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
        for line in lines:
            parts = line.strip().split(' ')
            if parts[0] == "VERTEX_SE3:QUAT":
                pose_id = int(parts[1])
                x, y, z = map(float, parts[2:5])
                qx, qy, qz, qw = map(float, parts[5:9])
                data['poses'].append(Pose3(pose_id, x, y, z, qx, qy, qz, qw))
            elif parts[0] == "EDGE_SE3:QUAT":
                id1, id2 = map(int, parts[1:3])
                x, y, z = map(float, parts[3:6])
                qx, qy, qz, qw = map(float, parts[6:10])
                info_vector = np.array(parts[10:], dtype=float)
                
                # Reconstruct the 6x6 information matrix from its upper-triangular form
                info_matrix = np.zeros((6, 6))
                indices_upper = np.triu_indices(6)
                info_matrix[indices_upper] = info_vector
                # Fill in the lower part to make the matrix symmetric
                indices_lower = (indices_upper[1], indices_upper[0])
                info_matrix[indices_lower] = info_vector
                
                data['edges'].append(Edge3(id1, id2, x, y, z, qx, qy, qz, qw, info_matrix))
    
        # ...

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    
    return data  

def gn_3d(data):
    poses = data['poses']
    edges = data['edges']

    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # Initialize factor graph and values
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    result = gtsam.Values()
    
    for pose in poses:
        r = gtsam.Rot3.Quaternion(pose.qw, pose.qx, pose.qy, pose.qz)
        t = gtsam.Point3(pose.x, pose.y, pose.z)
        initial.insert(pose.id, gtsam.Pose3(r, t))
        
    
    # Prior noise model
    prior_noise = gtsam.noiseModel.Diagonal.Variances(first_pose_prior_cov)
    graph.add(gtsam.PriorFactorPose3(0, initial.atPose3(0), prior_noise))

    for edge in edges:
        R = gtsam.Rot3.Quaternion(edge.qw, edge.qx, edge.qy, edge.qz)
        T = gtsam.Point3(edge.x, edge.y, edge.z)
        info = np.linalg.pinv(edge.info)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(info)
        graph.add(gtsam.BetweenFactorPose3(edge.id1, edge.id2, gtsam.Pose3(R,T),noise_model))

    
    params = gtsam.GaussNewtonParams()
    
    # Optimize using Gauss-Newton algorithm
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
    result = optimizer.optimize()


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
    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(0.1)
    # params.setRelinearizeSkip(10)
    isam = gtsam.ISAM2(params)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(first_pose_prior_cov)
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    # ...

    for pose in poses:

        frame_id = pose.id

        # create an empty factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        
        if frame_id==0:
            # initialization
            frame_id, x, y, z, *quat_v = pose
            frame_id = np.array(frame_id)
            frame_id = frame_id.astype(np.int32)
            # Notice that the order of quaternion is different from
            # the one in the g2o file. GTSAM uses (qw, qx, qy, qz).

            rot_mat = gtsam.Rot3(quat_v[-1], *quat_v[:3])
            t_vec = gtsam.Point3(x, y, z)
            graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(rot_mat, t_vec), prior_noise))
            initial_estimate.insert(frame_id, gtsam.Pose3(rot_mat, t_vec))
        
            pass
        else:
            # optimize new frame
            prev_pose = result.atPose3(frame_id - 1)
            initial_estimate.insert(frame_id, prev_pose)
            for edge in edges:
                if int(edge[1]) == frame_id:
                    id_e1, id_e2, dx, dy, dz, qx, qy, qz, qw, info = edge
                    id_e1 = np.array(id_e1).astype(np.int32)
                    id_e2 = np.array(id_e2).astype(np.int32)
                    
                    rot_mat = gtsam.Rot3(qw, qx, qy, qz)
                    t_vec = gtsam.Point3(dx, dy, dz)
                    
                    info_m = info
                    noise_model = gtsam.noiseModel.Gaussian.Information(info_m)
                    graph.add(gtsam.BetweenFactorPose3(id_e1, id_e2 , gtsam.Pose3(rot_mat, t_vec), noise_model))
            pass

        # update isam
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()
        graph.resize(0)
        initial_estimate.clear()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose3(result)
