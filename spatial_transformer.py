"""
Implementation of Spatial Transformer Networks

References
----------
[1] Spatial Transformer Networks
    Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
    Submitted on 5 Jun 2015

[2] https://github.com/tensorflow/models/tree/master/transformer/transformerlayer.py

[3] https://github.com/daviddao/spatial-transformer-tensorflow

[4] https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

[5] https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py

[6]  Fred L. Bookstein (1989):
     Principal warps: thin-plate splines and the decomposition of deformations. 
     IEEE Transactions on Pattern Analysis and Machine Intelligence.
     http://doi.org/10.1109/34.24792

"""

import tensorflow as tf
import math


"""
Legacy Function

"""
def transformer(inp, theta, out_size, name='SpatialTransformer', **kwargs):
    #with tf.variable_scope(name):
    stl = AffineTransformer(out_size)
    output = stl.transform(inp, theta, out_size)
    return output

class AffineVolumeTransformer(object):
    """Spatial Affine Volume Transformer Layer
    Implements a spatial transformer layer for volumetric 3D input.
    Implemented by Daniyar Turmukhambetov.
    """

    def __init__(self, out_size, name='SpatialAffineVolumeTransformer', interp_method='bilinear', **kwargs):
        """
        Parameters
        ----------
        out_size : tuple of three ints
            The size of the output of the spatial network (depth, height, width), i.e. z, y, x
        name : string
            The scope name of the variables in this network.

        """
        self.name = name
        self.out_size = out_size
        self.param_dim = 3*4
        self.interp_method=interp_method

        #with tf.variable_scope(self.name):
        self.voxel_grid = _meshgrid3d(self.out_size)
        
    
    def transform(self, inp, theta):
        """
        Affine Transformation of input tensor inp with parameters theta

        Parameters
        ----------
        inp : float
            The input tensor should have the shape 
            [batch_size, depth, height, width, in_channels].
        theta: float
            The output of the localisation network
            should have the shape
            [batch_size, 12].
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to :
            identity = np.array([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.]])
            identity = identity.flatten()
            theta = tf.Variable(initial_value=identity)

        """
        
        x_s, y_s, z_s = self._transform(inp, theta)

        output = _interpolate3d(
            inp, x_s, y_s, z_s,
            self.out_size,
            method=self.interp_method
            )

        batch_size, _, _, _, num_channels = inp.get_shape().as_list()
        output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], self.out_size[2], num_channels])

        return output


    
    def _transform(self, inp, theta):
        
        batch_size, _, _, _, num_channels = inp.get_shape().as_list()

        theta = tf.reshape(theta, (-1, 3, 4))
        voxel_grid = tf.tile(self.voxel_grid, [batch_size])
        voxel_grid = tf.reshape(voxel_grid, [batch_size, 4, -1])

        # Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
        T_g = tf.matmul(theta, voxel_grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])

        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        z_s_flat = tf.reshape(z_s, [-1])
        return x_s_flat, y_s_flat, z_s_flat
    

class AffineTransformer(object):
    """Spatial Affine Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and [3]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, name='SpatialAffineTransformer', interp_method='bilinear', **kwargs):
        """
        Parameters
        ----------
        out_size : tuple of two ints
            The size of the output of the spatial network (height, width).
        name : string
            The scope name of the variables in this network.

        """
        self.name = name
        self.out_size = out_size
        self.param_dim = 6
        self.interp_method=interp_method

        #with tf.variable_scope(self.name):
        self.pixel_grid = _meshgrid(self.out_size)
        
    
    def transform(self, inp, theta):
        """
        Affine Transformation of input tensor inp with parameters theta

        Parameters
        ----------
        inp : float
            The input tensor should have the shape 
            [batch_size, height, width, num_channels].
        theta: float
            The output of the localisation network
            should have the shape
            [batch_size, 6].
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to :
            identity = np.array([[1., 0., 0.],
                                 [0., 1., 0.]])
            identity = identity.flatten()
            theta = tf.Variable(initial_value=identity)

        """
        #with tf.variable_scope(self.name):
        x_s, y_s = self._transform(inp, theta)

        output = _interpolate(
            inp, x_s, y_s,
            self.out_size,
            method=self.interp_method
            )

        batch_size, _, _, num_channels = inp.get_shape().as_list()
        output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])

        return output


    
    def _transform(self, inp, theta):
        #with tf.variable_scope(self.name + '_affine_transform'):
        batch_size, _, _, num_channels = inp.get_shape().as_list()

        theta = tf.reshape(theta, (-1, 2, 3))
        pixel_grid = tf.tile(self.pixel_grid, [batch_size])
        pixel_grid = tf.reshape(pixel_grid, [batch_size, 3, -1])

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, pixel_grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        return x_s_flat, y_s_flat
    
class ProjectiveTransformer(object):
    """Spatial Projective Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and [3]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, name='SpatialProjectiveTransformer', interp_method='bilinear', **kwargs):   #interp_method='nearest' can be only used in inference mode
        """
        Parameters
        ----------
        out_size : tuple of two ints
            The size of the output of the spatial network (height, width).
        name : string
            The scope name of the variables in this network.

        """
        self.name = name
        self.out_size = out_size
        self.param_dim = 8
        self.interp_method=interp_method

        #with tf.variable_scope(self.name):
        self.pixel_grid = _meshgrid(self.out_size)
        
    
    def transform(self, inp, theta):
        """
        Projective Transformation of input tensor inp with parameters theta

        Parameters
        ----------
        inp : float
            The input tensor should have the shape 
            [batch_size, height, width, num_channels].
        theta: float
            The output of the localisation network
            should have the shape
            [batch_size, 8].
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to :
            identity = np.array([1., 0., 0.,
                                [0., 1., 0.,
                                [0., 0.])
            theta = tf.Variable(initial_value=identity)

        """
        #with tf.variable_scope(self.name):
        x_s, y_s = self._transform(inp, theta)

        output = _interpolate(
            inp, x_s, y_s,
            self.out_size,
            method=self.interp_method
            )

        batch_size, _, _, num_channels = inp.get_shape().as_list()
        output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])

        return output


    
    def _transform(self, inp, theta):
        #with tf.variable_scope(self.name + '_projective_transform'):
        batch_size, _, _, num_channels = inp.get_shape().as_list()

        theta = tf.reshape(theta, (batch_size, 8))
        theta = tf.concat([theta, tf.ones([batch_size, 1])], 1)
        theta = tf.reshape(theta, (batch_size, 3, 3))

        pixel_grid = tf.tile(self.pixel_grid, [batch_size])
        pixel_grid = tf.reshape(pixel_grid, [batch_size, 3, -1])
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, pixel_grid)

        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
        
        
        z_s += 0.0000001

        x_s = x_s/z_s
        y_s = y_s/z_s

        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        
        return x_s_flat, y_s_flat

class ElasticTransformer(object):
    """Spatial Elastic Transformer Layer with Thin Plate Spline deformations

    Implements a spatial transformer layer as described in [1]_.
    Based on [4]_ and [5]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, param_dim=2*16, name='SpatialElasticTransformer', interp_method='bilinear', **kwargs):
        """
        Parameters
        ----------
        out_size : tuple of two ints
            The size of the output of the spatial network (height, width).
        param_dim: int
            The 2 x number of control points that define 
            Thin Plate Splines deformation field. 
            number of control points *MUST* be a square of an integer. 
            2 x 16 by default.
        name : string
            The scope name of the variables in this network.
            
        """
        num_control_points = int(param_dim/2)
        assert param_dim == 2*num_control_points, 'param_dim must be 2 times a square of an integer.'

        self.name = name
        self.param_dim = param_dim
        self.interp_method=interp_method
        self.num_control_points = num_control_points
        self.out_size = out_size

        self.grid_size = math.floor(math.sqrt(self.num_control_points))
        assert self.grid_size*self.grid_size == self.num_control_points, 'num_control_points must be a square of an int'

        #with tf.variable_scope(self.name):
        # Create source grid
        self.source_points = ElasticTransformer.get_meshgrid(self.grid_size, self.grid_size)
        # Construct pixel grid
        self.pixel_grid = ElasticTransformer.get_meshgrid(self.out_size[1], self.out_size[0])
        self.num_pixels = self.out_size[0]*self.out_size[1]
        self.pixel_distances, self.L_inv =  self._initialize_tps(self.source_points, self.pixel_grid)


    def transform(self, inp, theta, forward=True, **kwargs):
        """
        Parameters
        ----------
        inp : float
            The input tensor should have the shape 
            [batch_size, height, width, num_channels].
        theta: float 
            Should have the shape of [batch_size, self.num_control_points x 2]
            Theta is the output of the localisation network, so it is 
            the x and y offsets of the destination coordinates 
            of each of the control points.
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to zeros:
            identity = np.zeros(16*2)
            identity = identity.flatten()
            theta = tf.Variable(initial_value=identity)

        """
        #with tf.variable_scope(self.name):
        # reshape destination offsets to be (batch_size, 2, num_control_points)
        # and add to source_points
        source_points = tf.expand_dims(self.source_points, 0)
        theta = source_points + tf.reshape(theta, [-1, 2, self.num_control_points])

        x_s, y_s = self._transform(
                inp, theta, self.num_control_points, 
                self.pixel_grid, self.num_pixels, 
                self.pixel_distances, self.L_inv, 
                self.name + '_elastic_transform', forward)
        if forward:
            output = _interpolate(
                inp, x_s, y_s,
                self.out_size,
                method=self.interp_method
                )
        else:
            rx_s, ry_s = self._transform(
                    inp, theta, self.num_control_points, 
                    self.pixel_grid, self.num_pixels, 
                    self.pixel_distances, self.L_inv, 
                    self.name + '_elastic_transform', forward)
            output = _interpolate(
                inp, rx_s, ry_s,
                self.out_size,
                method=self.interp_method
                )
            pass

    
        batch_size, _, _, num_channels = inp.get_shape().as_list()
        output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])
        return output

    def _transform(self, inp, theta, num_control_points, pixel_grid, num_pixels, pixel_distances, L_inv, name, forward=True):
        #with tf.variable_scope(name):
        batch_size = inp.get_shape().as_list()[0]

        # Solve as in ref [2]
        theta = tf.reshape(theta, [-1, num_control_points])
        coefficients = tf.matmul(theta, L_inv)
        coefficients = tf.reshape(coefficients, [-1, 2, num_control_points+3])

        # Transform each point on the target grid (out_size)
        right_mat = tf.concat([pixel_grid, pixel_distances], 0)
        right_mat = tf.tile(tf.expand_dims(right_mat, 0), (batch_size, 1, 1))
        transformed_points = tf.matmul(coefficients, right_mat)
        transformed_points = tf.reshape(transformed_points, [-1, 2, num_pixels])

        x_s_flat = tf.reshape(transformed_points[:,0,:], [-1])
        y_s_flat = tf.reshape(transformed_points[:,1,:], [-1])

        return x_s_flat, y_s_flat

    # U function for the new point and each source point 
    @staticmethod
    def U_func(points1, points2):
        # The U function is simply U(r) = r^2 * log(r^2), as in ref [5]_,
        # where r is the euclidean distance
        r_sq  = tf.transpose(tf.reduce_sum(tf.square(points1 - points2), axis=0))
        log_r = tf.log(r_sq)
        log_r = tf.where(tf.is_inf(log_r), tf.zeros_like(log_r), log_r)
        phi = r_sq * log_r

        # The U function is simply U(r) = r, where r is the euclidean distance
        #phi = tf.sqrt(r_sq)
        return phi

    
    @staticmethod
    def get_meshgrid(grid_size_x, grid_size_y):
        # Create 2 x num_points array of source points
        x_points, y_points = tf.meshgrid(
            tf.linspace(-1.0, 1.0, int(grid_size_x)),
            tf.linspace(-1.0, 1.0, int(grid_size_y)))
        x_flat = tf.reshape(x_points, (1,-1))
        y_flat = tf.reshape(y_points, (1,-1))
        points = tf.concat([x_flat, y_flat], 0)
        return points


    def _initialize_tps(self, source_points, pixel_grid):
        """
        Initializes the thin plate spline calculation by creating the source
        point array and the inverted L matrix used for calculating the
        transformations as in ref [5]_
    
        Returns
        ----------
        right_mat : float
            Tensor of shape [num_control_points + 3, out_height*out_width].
        L_inv : float
            Tensor of shape [num_control_points + 3, num_control_points]. 
        source_points : float
            Tensor of shape (2, num_control_points).

        """
    
        tL = ElasticTransformer.U_func(tf.expand_dims(source_points, 2), tf.expand_dims(source_points, 1))

        # Initialize L 
        L_top = tf.concat([tf.zeros([2,3]), source_points], 1)
        L_mid = tf.concat([tf.zeros([1, 2]), tf.ones([1, self.num_control_points+1])], 1)
        L_bot = tf.concat([tf.transpose(source_points), tf.ones([self.num_control_points, 1]), tL], 1)

        L = tf.concat([L_top, L_mid, L_bot], 0)
        L_inv = tf.matrix_inverse(L)
    
        # Construct right mat
        to_transform = tf.expand_dims(pixel_grid, 2)
        stacked_source_points = tf.expand_dims(source_points, 1)
        distances = ElasticTransformer.U_func(to_transform, stacked_source_points)
    
        # Add in the coefficients for the affine translation (1, x, and y,
        # corresponding to a_1, a_x, and a_y)
        ones = tf.ones(shape=[1, self.num_pixels])
        pixel_distances = tf.concat([ones, distances], 0)
        L_inv = tf.transpose(L_inv[:,3:])

        return pixel_distances, L_inv



"""
Common Functions

"""

def _meshgrid3d(out_size):
    """
    the regular grid of coordinates to sample the values after the transformation
    
    """
    #with tf.variable_scope('meshgrid3d'):

    # This should be equivalent to:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    #z_t, y_t, x_t = tf.meshgrid(tf.linspace(0., out_size[0]-1.,  out_size[0]),
    #                       tf.linspace(0., out_size[1]-1.,  out_size[1]), 
    #                       tf.linspace(0., out_size[2]-1.,  out_size[2]), indexing='ij')

    z_t, y_t, x_t = tf.meshgrid(tf.linspace(-1., 1.,  out_size[0]),
                           tf.linspace(-1., 1.,  out_size[1]), 
                           tf.linspace(-1., 1.,  out_size[2]), indexing='ij')

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    z_t_flat = tf.reshape(z_t, (1, -1))

    ones = tf.ones_like(x_t_flat)
    grid = tf.concat([x_t_flat, y_t_flat, z_t_flat, ones], 0)
    grid = tf.reshape(grid, [-1])
    return grid

def _meshgrid(out_size):
    """
    the regular grid of coordinates to sample the values after the transformation
    
    """
    #with tf.variable_scope('meshgrid'):

    # This should be equivalent to:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0,  out_size[1]),
                           tf.linspace(-1.0, 1.0,  out_size[0]))
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))


    grid = tf.concat([x_t_flat, y_t_flat, tf.ones_like(x_t_flat)], 0)
    grid = tf.reshape(grid, [-1])

    return grid


def _repeat(x, n_repeats):
    #with tf.variable_scope('_repeat'):
    rep = tf.tile(tf.expand_dims(x,1), [1, n_repeats])
    return tf.reshape(rep, [-1])

def _interpolate(im, x, y, out_size, method):
    if method=='bilinear':
        return bilinear_interp(im, x, y, out_size)
    if method=='bicubic':
        return bicubic_interp(im, x, y, out_size)
    if method=='nearest':
        return Nearest_interp(im, x, y, out_size)
    return None

def _interpolate3d(vol, x, y, z, out_size, method='bilinear'):
    return bilinear_interp3d(vol, x, y, z, out_size)

def bilinear_interp3d(vol, x, y, z, out_size, edge_size=1):
    #with tf.variable_scope('bilinear_interp3d'):
    batch_size, depth, height, width, channels = vol.get_shape().as_list()

    if edge_size>0:
        vol = tf.pad(vol, [[0,0], [edge_size,edge_size], [edge_size,edge_size], [edge_size,edge_size], [0,0]], mode='CONSTANT')

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.float32)

    depth_f  = tf.cast(depth, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f  = tf.cast(width, tf.float32)

    out_depth  = out_size[0]
    out_height = out_size[1]
    out_width  = out_size[2]

    # scale indices to [0, width/height/depth - 1]
    x = (x + 1.) / 2. * (width_f  -1.)
    y = (y + 1.) / 2. * (height_f -1.)
    z = (z + 1.) / 2. * (depth_f  -1.)

    # clip to to [0, width/height/depth - 1] +- edge_size
    x = tf.clip_by_value(x, -edge_size, width_f  -1. + edge_size)
    y = tf.clip_by_value(y, -edge_size, height_f -1. + edge_size)
    z = tf.clip_by_value(z, -edge_size, depth_f  -1. + edge_size)

    x += edge_size
    y += edge_size
    z += edge_size

    # do sampling
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    z0_f = tf.floor(z)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    z1_f = z0_f + 1

    x0 = tf.cast(x0_f, tf.int32)
    y0 = tf.cast(y0_f, tf.int32)
    z0 = tf.cast(z0_f, tf.int32)

    x1 = tf.cast(tf.minimum(x1_f, width_f  - 1. + 2*edge_size),  tf.int32)
    y1 = tf.cast(tf.minimum(y1_f, height_f - 1. + 2*edge_size), tf.int32)
    z1 = tf.cast(tf.minimum(z1_f, depth_f  - 1. + 2*edge_size), tf.int32)

    dim3 = (width + 2*edge_size)
    dim2 = (width + 2*edge_size)*(height + 2*edge_size)
    dim1 = (width + 2*edge_size)*(height + 2*edge_size)*(depth + 2*edge_size)

    base = _repeat(tf.range(batch_size)*dim1, out_depth*out_height*out_width)
    base_z0 = base + z0*dim2
    base_z1 = base + z1*dim2

    base_y00 = base_z0 + y0*dim3
    base_y01 = base_z0 + y1*dim3
    base_y10 = base_z1 + y0*dim3
    base_y11 = base_z1 + y1*dim3

    idx_000 = base_y00 + x0
    idx_001 = base_y00 + x1
    idx_010 = base_y01 + x0
    idx_011 = base_y01 + x1
    idx_100 = base_y10 + x0
    idx_101 = base_y10 + x1
    idx_110 = base_y11 + x0
    idx_111 = base_y11 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    vol_flat = tf.reshape(vol, [-1, channels])
    I000 = tf.gather(vol_flat, idx_000)
    I001 = tf.gather(vol_flat, idx_001)
    I010 = tf.gather(vol_flat, idx_010)
    I011 = tf.gather(vol_flat, idx_011)
    I100 = tf.gather(vol_flat, idx_100)
    I101 = tf.gather(vol_flat, idx_101)
    I110 = tf.gather(vol_flat, idx_110)
    I111 = tf.gather(vol_flat, idx_111)

    # and finally calculate interpolated values
    w000 = tf.expand_dims((z1_f-z)*(y1_f-y)*(x1_f-x),1)
    w001 = tf.expand_dims((z1_f-z)*(y1_f-y)*(x-x0_f),1)
    w010 = tf.expand_dims((z1_f-z)*(y-y0_f)*(x1_f-x),1)
    w011 = tf.expand_dims((z1_f-z)*(y-y0_f)*(x-x0_f),1)
    w100 = tf.expand_dims((z-z0_f)*(y1_f-y)*(x1_f-x),1)
    w101 = tf.expand_dims((z-z0_f)*(y1_f-y)*(x-x0_f),1)
    w110 = tf.expand_dims((z-z0_f)*(y-y0_f)*(x1_f-x),1)
    w111 = tf.expand_dims((z-z0_f)*(y-y0_f)*(x-x0_f),1)

    output = tf.add_n([
        w000*I000, 
        w001*I001, 
        w010*I010, 
        w011*I011, 
        w100*I100, 
        w101*I101, 
        w110*I110, 
        w111*I111])
    return output


def bilinear_interp(im, x, y, out_size):
    #with tf.variable_scope('bilinear_interp'):
    batch_size, height, width, channels = im.get_shape().as_list()

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    out_height = out_size[0]
    out_width = out_size[1]

    # scale indices from [-1, 1] to [0, width/height - 1]
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y, -1, 1)
    x = (x + 1.0) / 2.0 * (width_f-1.0)
    y = (y + 1.0) / 2.0 * (height_f-1.0)

    # do sampling
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    x0 = tf.cast(x0_f, tf.int32)
    y0 = tf.cast(y0_f, tf.int32)
    x1 = tf.cast(tf.minimum(x1_f, width_f - 1),  tf.int32)
    y1 = tf.cast(tf.minimum(y1_f, height_f - 1), tf.int32)

    dim2 = width
    dim1 = width*height
   
    base = _repeat(tf.range(batch_size)*dim1, out_height*out_width)

    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2

    idx_00 = base_y0 + x0
    idx_01 = base_y0 + x1
    idx_10 = base_y1 + x0
    idx_11 = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, [-1, channels])

    I00 = tf.gather(im_flat, idx_00)
    I01 = tf.gather(im_flat, idx_01)
    I10 = tf.gather(im_flat, idx_10)
    I11 = tf.gather(im_flat, idx_11)

    # and finally calculate interpolated values
    w00 = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
    w01 = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
    w10 = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
    w11 = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)

    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])
    return output


def bicubic_interp(im, x, y, out_size):
    alpha = -0.75 # same as in tf.image.resize_images, see:
    #  tensorflow/tensorflow/core/kernels/resize_bicubic_op.cc
    bicubic_coeffs = (
            (1, 0,     -(alpha+3), (alpha+2)),
            (0, alpha, -2*alpha,   alpha    ),
            (0, -alpha, 2*alpha+3, -alpha-2 ),
            (0, 0,      alpha,     -alpha   )
            )

    #with tf.variable_scope('bicubic_interp'):
    batch_size, height, width, channels = im.get_shape().as_list()

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    out_height = out_size[0]
    out_width = out_size[1]

    # scale indices from [-1, 1] to [0, width/height - 1]
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y, -1, 1)
    x = (x + 1.0) / 2.0 * (width_f-1.0)
    y = (y + 1.0) / 2.0 * (height_f-1.0)

    # do sampling
    # integer coordinates of 4x4 neighbourhood around (x0_f, y0_f)
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    xm1_f = x0_f - 1
    ym1_f = y0_f - 1
    xp1_f = x0_f + 1
    yp1_f = y0_f + 1
    xp2_f = x0_f + 2
    yp2_f = y0_f + 2

    # clipped integer coordinates
    xs = [0, 0, 0, 0]
    ys = [0, 0, 0, 0]
    xs[0] = tf.cast(x0_f, tf.int32)
    ys[0] = tf.cast(y0_f, tf.int32)
    xs[1] = tf.cast(tf.maximum(xm1_f, 0), tf.int32)
    ys[1] = tf.cast(tf.maximum(ym1_f, 0), tf.int32)
    xs[2] = tf.cast(tf.minimum(xp1_f, width_f - 1),  tf.int32)
    ys[2] = tf.cast(tf.minimum(yp1_f, height_f - 1), tf.int32)
    xs[3] = tf.cast(tf.minimum(xp2_f, width_f - 1),  tf.int32)
    ys[3] = tf.cast(tf.minimum(yp2_f, height_f - 1), tf.int32)

    # indices of neighbours for the batch
    dim2 = width
    dim1 = width*height
    base = _repeat(tf.range(batch_size)*dim1, out_height*out_width)

    idx = []
    for i in range(4):
        idx.append([])
        for j in range(4):
            cur_idx = base + ys[i]*dim2 + xs[j]
            idx[i].append(cur_idx)

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, [-1, channels])

    Is = []
    for i in range(4):
        Is.append([])
        for j in range(4):
            Is[i].append(tf.gather(im_flat, idx[i][j]))

    def get_weights(x, x0_f):
        tx = (x-x0_f)
        tx2 = tx * tx
        tx3 = tx2 * tx
        t = [1, tx, tx2, tx3]
        weights = []
        for i in range(4):
            result = 0
            for j in range(4):
                result = result + bicubic_coeffs[i][j]*t[j]
            result = tf.reshape(result, [-1, 1])
            weights.append(result)
        return weights


    # to calculate interpolated values first, 
    # interpolate in x dim 4 times for y=[0, -1, 1, 2]
    weights = get_weights(x, x0_f)
    x_interp = []
    for i in range(4):
        result = []
        for j in range(4):
            result = result + [weights[j]*Is[i][j]]
        x_interp.append(tf.add_n(result))

    # finally, interpolate in y dim using interpolations in x dim
    weights = get_weights(y, y0_f)
    y_interp = []
    for i in range(4):
        y_interp = y_interp + [weights[i]*x_interp[i]]

    output = tf.add_n(y_interp)
    return output
    
    
def Nearest_interp(im, x, y, out_size):
    #with tf.variable_scope('bilinear_interp'):
    batch_size, height, width, channels = im.get_shape().as_list()

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    out_height = out_size[0]
    out_width = out_size[1]

    # scale indices from [-1, 1] to [0, width/height - 1]
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y, -1, 1)
    x = (x + 1.0) / 2.0 * (width_f-1.0)
    y = (y + 1.0) / 2.0 * (height_f-1.0)
    
    # do sampling
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)


    x0 = tf.cast(x0_f, tf.int32)
    y0 = tf.cast(y0_f, tf.int32)
    
    dim2 = width
    dim1 = width*height
   
    base = _repeat(tf.range(batch_size)*dim1, out_height*out_width)

    base_y0 = base + y0*dim2

    idx = base_y0 + x0

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, [-1, channels])

    I00 = tf.gather(im_flat, idx)
    
    output = I00
    return output
