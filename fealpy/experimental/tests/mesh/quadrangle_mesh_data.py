import numpy as np

init_mesh_data = [
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
        "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "NN": 4,
        "NE": 4,
        "NF": 4,
        "NC": 1,
        "face2cell": np.array([[0, 0, 0, 0],
                               [0, 0, 3, 3],
                               [0, 0, 1, 1],
                               [0, 0, 2, 2]], dtype=np.int32),
    }, ]
box_data = [
    {
        "box": [0, 1, 0, 1],
        "nx": 1,
        "ny": 1,
        "node": np.array([[0., 0.],
                          [0., 1.],
                          [1., 0.],
                          [1., 1.]], dtype=np.float64),
        "edge": np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
        "cell": np.array([[0, 2, 3, 1]], dtype=np.int32),
        "NN": 4,
        "NE": 4,
        "NF": 4,
        "NC": 1,
        "face2cell": np.array([[0, 0, 3, 3],
                               [0, 0, 0, 0],
                               [0, 0, 2, 2],
                               [0, 0, 1, 1]], dtype=np.int32),
    }, ]
entity_data = [
    {
        "entity_measure": (np.array([0.0], dtype=np.float64),
                           np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                           np.array([1.0], dtype=np.float64)),
        "bcs": (np.array([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=np.float64),
                np.array([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=np.float64)),
        "ws": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
        "point": np.array([[[0.21132487, 0.21132487]],
                           [[0.21132487, 0.78867513]],
                           [[0.78867513, 0.21132487]],
                           [[0.78867513, 0.78867513]]], dtype=np.float64),
    }, ]
geo_data = [
    {
        "edge_frame": (np.array([[0., -1.], [-1., 0.], [1., 0.], [0., 1.]], dtype=np.float64),
                       np.array([[1., 0.], [0., -1.], [0., 1.], [-1., 0.]], dtype=np.float64)),
        "edge_unit_normal": np.array([[0., -1.], [-1., 0.], [1., 0.], [0., 1.]], dtype=np.float64),
    }, ]
cal_data = [
    {
        "shape_function": np.array([[0.62200847, 0.16666667, 0.16666667, 0.0446582],
                                    [0.16666667, 0.62200847, 0.0446582, 0.16666667],
                                    [0.16666667, 0.0446582, 0.62200847, 0.16666667],
                                    [0.0446582, 0.16666667, 0.16666667, 0.62200847]],
                                   dtype=np.float64),
        "grad_shape_function": np.array(
            [[[-0.78867513, -0.78867513], [-0.21132487, 0.78867513], [0.78867513, -0.21132487],
              [0.21132487, 0.21132487]],
             [[-0.21132487, -0.78867513], [-0.78867513, 0.78867513], [0.21132487, -0.21132487],
              [0.78867513, 0.21132487]],
             [[-0.78867513, -0.21132487], [-0.21132487, 0.21132487], [0.78867513, -0.78867513],
              [0.21132487, 0.78867513]],
             [[-0.21132487, -0.21132487], [-0.78867513, 0.21132487], [0.21132487, -0.78867513],
              [0.78867513, 0.78867513]]],
            dtype=np.float64),
        "grad_shape_function_x": np.array(
            [[[[-0.78867513, - 0.78867513], [-0.21132487, 0.78867513], [0.78867513, - 0.21132487],
               [0.21132487, 0.21132487]]],
             [[[-0.21132487, - 0.78867513], [-0.78867513, 0.78867513], [0.21132487, -0.21132487],
               [0.78867513, 0.21132487]]],
             [[[-0.78867513, - 0.21132487], [-0.21132487, 0.21132487], [0.78867513, - 0.78867513],
               [0.21132487, 0.78867513]]],
             [[[-0.21132487, - 0.21132487], [-0.78867513, 0.21132487], [0.21132487, - 0.78867513],
               [0.78867513, 0.78867513]]]],
            dtype=np.float64),
        "jacobi_matrix": np.array(
            [[[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]]],
            dtype=np.float64),
        "first_fundamental_form": np.array(
            [[[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]]],
            dtype=np.float64),
    }, ]
extend_data = [
    {
        "p": 2,
        "number_of_local_ipoints": 9,
        "number_of_global_ipoints": 9,
        "number_of_corner_nodes": 4,
        "interpolation_points": np.array([[0., 0.],
                                          [1., 0.],
                                          [1., 1.],
                                          [0., 1.],
                                          [0.5, 0.],
                                          [0., 0.5],
                                          [1., 0.5],
                                          [0.5, 1.],
                                          [0.5, 0.5]], dtype=np.float64),
        "p0": 1,
        "p1": 3,
        "prolongation_matrix": np.array([[1., 0., 0., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.],
                                         [0.66666667, 0.33333333, 0., 0.],
                                         [0.33333333, 0.66666667, 0., 0.],
                                         [0.33333333, 0., 0., 0.66666667],
                                         [0.66666667, 0., 0., 0.33333333],
                                         [0., 0.66666667, 0.33333333, 0.],
                                         [0., 0.33333333, 0.66666667, 0.],
                                         [0., 0., 0.66666667, 0.33333333],
                                         [0., 0., 0.33333333, 0.66666667],
                                         [0.44444444, 0.22222222, 0.11111111, 0.22222222],
                                         [0.22222222, 0.11111111, 0.22222222, 0.44444444],
                                         [0.22222222, 0.44444444, 0.22222222, 0.11111111],
                                         [0.11111111, 0.22222222, 0.44444444, 0.22222222]], dtype=np.float64),
        "cell_to_ipoint": np.array([[0, 5, 3, 4, 8, 7, 1, 6, 2]], dtype=np.int32),
        "jacobi_at_corner": np.array([[1., 1., 1., 1.]], dtype=np.float64),
        "angle": np.array([[1.57079633, 1.57079633, 1.57079633, 1.57079633]], dtype=np.float64),
        "cell_quality": np.array([1.], np.float64),
    }, ]
uniform_refine_data = [{
    "t": 0,
}, ]