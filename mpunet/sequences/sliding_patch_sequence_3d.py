from . import PatchSequence3D
from mpunet.interpolation.linalg import mgrid_to_points
import numpy as np


def standardize_strides(strides):
    if isinstance(strides, list):
        return tuple(strides)
    elif isinstance(strides, tuple):
        return strides
    else:
        return 3 * (int(strides),)
#
# class SlidingPatchSequence3D():
#
#     def __init__(self, ):
#         self._dim_r = None
#
#     @property
#     def dim_r(self):
#         return self._dim_r
#
#     @dim_r.setter
#     def dim_r(self, value):
#         if value < 0:
#             raise ValueError("Invalid dim_r size of %i" % value)
#         self._dim_r = value
#         self.a = 1




class SlidingPatchSequence3D(PatchSequence3D):
    def __init__(self, strides=None, no_log=False, *args, **kwargs):

        """
        strides: tuple (s1, s2, s3) of strides or integer s --> (s, s, s)
        """
        super().__init__(no_log=True, *args, **kwargs)

        # Stride attribute gives the number pixel distance between
        # patch samples in 3 dimensions

        if strides is None:
            strides = self.dim//2

        self._dim_r = None
        # self.dim_r = None # original image  dimension
        # self.dim : patch dimension
        self.corners = None
        self.ind = None
        self.strides = standardize_strides(strides)

        if not self.is_validation and not no_log:
            self.log()


    @property
    def dim_r(self):
        return self._dim_r

    @dim_r.setter
    def dim_r(self, value):
        if np.sum(value) < 0:
            raise ValueError("Invalid dim_r size of %i" % value)
        self._dim_r = np.asarray(value)
        self.corners = self.get_patch_corners().astype(np.uint16)
        self.ind = np.arange(self.corners.shape[0])

    # @property
    def __len__(self):

        return super().__len__()

        if self.dim_r is None:
            one_shape = self.image_pair_queue.get_random_image_no_yield().shape[:3]
            n_patches = np.ceil((np.asarray(one_shape) - self.dim) / self.strides).astype(int)
            n_patches = np.prod(n_patches)//2

            """
            Note: Need to change this when the different image shapes various too much
             
            """

        else:
            n_patches = self.corners.shape[0]

        # breakpoint()

        return n_patches * len(self.image_pair_loader)

    def get_patch_corners(self):
        if self.dim_r is None:
            raise ValueError("Cannot call get patch corner before setting original dim")
        
        # breakpoint()
        
        n_patches = np.ceil((self.dim_r - self.dim)/self.strides).astype(int)

        # xc = np.linspace(0, self.dim_r[0], self.strides[0]).astype(int)
        # yc = np.linspace(0, self.dim_r[1], self.strides[1]).astype(int)
        # zc = np.linspace(0, self.dim_r[2], self.strides[2]).astype(int)

        # use np.rint ?????????????????????????

        xc = np.floor(np.linspace(0, self.dim_r[0], n_patches[0], endpoint=False)).astype(int)
        yc = np.floor(np.linspace(0, self.dim_r[1], n_patches[1], endpoint=False)).astype(int)
        zc = np.floor(np.linspace(0, self.dim_r[2], n_patches[2], endpoint=False)).astype(int)

        return mgrid_to_points(np.meshgrid(xc, yc, zc))

    def get_box_coords(self, _im):

        if self.dim_r is None:
            raise ValueError("Cannot call get patch corner before setting original dim")

        return self.corners[np.random.choice(self.ind)]

    def get_base_patches(self, image):
        # Get sliding windows for X
        X = image.image

        if self.dim_r is None:
            raise ValueError("Cannot call get patch corner before setting original dim")

        for xc, yc, zc in self.corners:
            
            # breakpoint()
            
            xc_upper_bound = min(xc + self.dim, self.dim_r[0])
            yc_upper_bound = min(yc + self.dim, self.dim_r[1])
            zc_upper_bound = min(zc + self.dim, self.dim_r[2])

            patch = X[xc:xc_upper_bound, yc:yc_upper_bound, zc:zc_upper_bound]

            yield image.scaler.transform(patch), (xc, yc, zc)

    def log(self):
        self.logger("Sequence Generator: %s" % self.__class__.__name__)
        self.logger("Box dimensions:     %s" % self.dim)
        # self.logger("Image dimensions:   %s" % self.im_dim)
        # self.logger("Sample space dim:   %s" % self.dim_r)
        self.logger("Strides:            %s" % list(self.strides))
        # self.logger("N boxes:            %s" % self.corners.shape[0])
        self.logger("Batch size:         %s" % self.batch_size)
        self.logger("N fg slices/batch:  %s" % self.n_fg_slices)
