from functools import partial
import numpy as np
import webbpsf_ext
from .winnie_jwst_fm import generate_nircam_psf_grid
from scipy.ndimage import map_coordinates
import scipy
from skimage.transform import resize

class WinniePSF:

    def __init__(self, psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs):
        self.psfs = np.array(psfs)
        self.psf_inds_rolls = np.array(psf_inds_rolls)
        self.im_mask_rolls = np.array(im_mask_rolls)
        self.psf_offsets = np.array(psf_offsets)
        self.psf_parangs = np.array(psf_parangs)

    def get_convolved_cube(self, image):
        convolved_cube = self.convolve_cube(image=image, cent=np.array((image.shape[1] / 2, image.shape[0] / 2)))
        return convolved_cube

    # Assuming parangs are already unique
    def convolve_cube(self, image, cent):
        nT = np.size(self.psf_parangs)
        ny, nx = image.shape[-2:]

        # Rotate the hypercube for each unique angle
        inp_rot_uni = self.rotate_hypercube(
            np.tile(image[None, ...], (nT, 1, 1)), self.psf_parangs, cent, cval0=0.
        )
        imcube = np.zeros((nT, ny, nx))

        # Convolve rotated images
        for i in range(0, np.size(self.psf_parangs)):
            imcube[i] = self.convolve_with_spatial_psfs(
                    inp_rot_uni[i],
                    psf_inds=self.psf_inds_rolls[i],
                    im_mask=self.im_mask_rolls[i],
                )
        return imcube

    def rotate_hypercube(self, hcube, angles, cent, cval0=0.):
        rot_hcube = np.empty_like(hcube)

        for i in range(0, np.size(angles)):
            rot_hcube[i] = rotate_image(hcube[i], angles[i], reshape=False, cval=cval0, order=1)

        return rot_hcube
    

    def convolve_with_spatial_psfs(self, image, psf_inds, im_mask):
        
        im_mask = resize(im_mask, image.shape, order=1, mode='edge', preserve_range=True)
        psf_inds = nearest_resize(psf_inds, image.shape)

        im = image * im_mask
        nonzero = im != 0
        psf_inds_masked = np.where(nonzero, psf_inds, -1)

        imcon = np.zeros_like(im)


        for i in np.unique(psf_inds_masked)[1:]:    #### QUESTIONABLE: pre-computing the number of unique psfs
            mask_i = psf_inds_masked == i
            im_to_convolve = np.where(mask_i, im, 0.0)
            imcon += scipy.signal.fftconvolve(im_to_convolve, self.psfs[i], mode='same')

        return imcon

def nearest_resize(array, new_shape):
    """Resize an array using nearest-neighbor sampling (pure NumPy)."""
    old_h, old_w = array.shape
    new_h, new_w = new_shape

    row_ratio = old_h / new_h
    col_ratio = old_w / new_w

    row_coords = (np.arange(new_h) * row_ratio + 0.5).astype(int)
    col_coords = (np.arange(new_w) * col_ratio + 0.5).astype(int)

    # Clamp indices to stay valid
    row_coords = np.clip(row_coords, 0, old_h - 1)
    col_coords = np.clip(col_coords, 0, old_w - 1)

    return array[row_coords[:, None], col_coords]

def rotate_image(image, angle, reshape=False, cval=0.0, order=1):
    
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # Compute the center of the image
    ny, nx = image.shape
    center_y, center_x = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Generate coordinate grid
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Shift coordinates to center
    x_shifted = x - center_x
    y_shifted = y - center_y

    # Apply rotation matrix
    x_rot = cos_theta * x_shifted + sin_theta * y_shifted + center_x
    y_rot = -sin_theta * x_shifted + cos_theta * y_shifted + center_y

    # Interpolate using map_coordinates
    rotated_image = map_coordinates(image, [y_rot.ravel(), x_rot.ravel()], order=order, cval=cval)
    rotated_image = rotated_image.reshape(image.shape)

    return rotated_image