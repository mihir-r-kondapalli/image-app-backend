import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.signal as ss
from .winnie_class import WinniePSF

class Jax_class:

    params = {}

    @classmethod
    def unpack_pars(cls, p_arr):
        """
        This function takes a parameter array (params) and unpacks it into a
        dictionary with the parameter names as keys.
        """
        p_dict = {}
        keys = list(cls.params.keys())
        i = 0
        for i in range(0, len(p_arr)):
            p_dict[keys[i]] = p_arr[i]

        return p_dict

    @classmethod
    def pack_pars(cls, p_dict):
        """
        This function takes a parameter dictionary and packs it into a JAX array
        where the order is set by the parameter name list defined on the class.
        """    
        p_arrs = []
        for name in cls.params.keys():
            p_arrs.append(p_dict[name])
        return np.asarray(p_arrs)


class DustEllipticalDistribution2PowerLaws(Jax_class):
    """
    """

    params = {'ain': 5., 'aout': -5., 'a': 60., 'e': 0., 'ksi0': 1.,'gamma': 2., 'beta': 1.,
                        'amin': 0., 'dens_at_r0': 1., 'accuracy': 5.e-3, 'zmax': 0., "p": 0., "rmax": 0.,
                        'pmin': 0., "apeak": 0., "apeak_surface_density": 0., "itiltthreshold": 0.}

    @classmethod
    def init(cls, accuracy=5.e-3, ain=5., aout=-5., a=60., e=0., ksi0=1., gamma=2., beta=1., amin=0., dens_at_r0=1.):
        """
        Constructor for the Dust_distribution class.

        We assume the dust density is 0 radially after it drops below 0.5%
        (the accuracy variable) of the peak density in
        the midplane, and vertically whenever it drops below 0.5% of the
        peak density in the midplane
        """

        p_dict = {}
        p_dict["accuracy"] = accuracy

        p_dict["ksi0"] = ksi0
        p_dict["gamma"] = gamma
        p_dict["beta"] = beta
        p_dict["zmax"] = ksi0*(-np.log(p_dict["accuracy"]))**(1./(gamma+1e-8))

        # Set Vertical Density Analogue
        gamma = np.where(gamma < 0., 0.1, gamma)
        ksi0 = np.where(ksi0 < 0., 0.1, ksi0)
        beta = np.where(beta < 0., 0., beta)

        # Set Radial Density Analogue
        ain = np.where(ain < 0.01, 0.01, ain)
        aout = np.where(aout > -0.01, -0.01, aout)
        e = np.where(e < 0., 0., e)
        e = np.where(e >= 1, 0.99, e)
        amin = np.where(amin < 0., 0., amin)
        dens_at_r0 = np.where(dens_at_r0 < 0., 0., dens_at_r0)

        p_dict["ain"] = ain
        p_dict["aout"] = aout
        p_dict["a"] = a
        p_dict["e"] = e
        p_dict["p"] = p_dict["a"]*(1-p_dict["e"]**2)
        p_dict["amin"] = amin
        # we assume the inner hole is also elliptic (convention)
        p_dict["pmin"] = p_dict["amin"]*(1-p_dict["e"]**2)
        p_dict["dens_at_r0"] = dens_at_r0

        # maximum distance of integration, AU
        p_dict["rmax"] = p_dict["a"]*p_dict["accuracy"]**(1/(p_dict["aout"]+1e-8))
        p_dict["apeak"] = p_dict["a"] * np.power(-p_dict["ain"]/(p_dict["aout"]+1e-8),
                                        1./(2.*(p_dict["ain"]-p_dict["aout"])))
        Gamma_in = np.abs(p_dict["ain"]+p_dict["beta"] + 1e-8)
        Gamma_out = -np.abs(p_dict["aout"]+p_dict["beta"] + 1e-8)
        p_dict["apeak_surface_density"] = p_dict["a"] * np.power(-Gamma_in/Gamma_out,
                                                        1./(2.*(Gamma_in-Gamma_out+1e-8)))
        # the above formula comes from Augereau et al. 1999.
        p_dict["itiltthreshold"] = np.rad2deg(np.arctan(p_dict["rmax"]/p_dict["zmax"]))

        return cls.pack_pars(p_dict)
    
    @classmethod
    def density_cylindrical(cls, distr_params, r, costheta, z):
        """ Returns the particule volume density at r, theta, z
        """
        distr = cls.unpack_pars(distr_params)

        radial_ratio = r*(1-distr["e"]*costheta)/((distr["p"])+1e-8)

        den = (np.power(np.abs(radial_ratio)+1e-8, -2*distr["ain"]) +
               np.power(np.abs(radial_ratio)+1e-8, -2*distr["aout"]))
        radial_density_term = np.sqrt(2./den+1e-8)*distr["dens_at_r0"]
        #if distr["pmin"] > 0:
        #    radial_density_term[r/(distr["pmin"]/(1-distr["e"]*costheta)) <= 1] = 0
        radial_density_term = np.where(distr["pmin"] > 0, 
                                        np.where(r*(1-distr["e"]*costheta)/((distr["p"])+1e-8) <= 1, 0., radial_density_term),
                                        radial_density_term)

        den2 = distr["ksi0"]*np.power(np.abs(radial_ratio+1e-8), distr["beta"]) + 1e-8
        vertical_density_term = np.exp(-np.power((np.abs(z)+1e-8)/(np.abs(den2+1e-8)), np.abs(distr["gamma"])+1e-8))
        return radial_density_term*vertical_density_term

class DoubleHenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a double Henyey
    Greenstein function.
    """

    params = {'g1': 0.5, 'g2': -0.3, 'weight': 0.7}

    @classmethod
    def init(cls, func_params):
        """
        """

        p_dict = {}
        p_dict['g1'] = func_params[0]
        p_dict['g2'] = func_params[1]
        p_dict['weight'] = func_params[2]

        return cls.pack_pars(p_dict)
    
    @classmethod
    def compute_phase_function_from_cosphi(cls, phase_func_params, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """

        p_dict = cls.unpack_pars(phase_func_params)

        hg1 = p_dict['weight'] * 1./(4*np.pi)*(1-p_dict["g1"]**2) / \
            (1+p_dict["g1"]**2-2*p_dict["g1"]*cos_phi)**(3./2.)
        hg2 = (1-p_dict['weight']) * 1./(4*np.pi)*(1-p_dict["g2"]**2) / \
            (1+p_dict["g2"]**2-2*p_dict["g2"]*cos_phi)**(3./2.)
        
        return hg1+hg2
    
    @classmethod
    def compute_phase_function_from_cosphi(cls, spline_model, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        spline_model : InterpolatedUnivariateSpline
            spline model to represent scattering light phase function
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        
        return spline_model(cos_phi)
    

class GAUSSIAN_PSF(Jax_class):

    params = {'FWHM': 3., 'xo': 0., 'yo': 0., 'theta': 0., 'offset': 0., 'amplitude': 1.}

    #define model function and pass independant variables x and y as a list
    @classmethod
    def generate(cls, image, psf_params):
        ny, nx = image.shape  # Get image size
        x = np.linspace(-nx // 2, nx // 2, nx)
        y = np.linspace(-ny // 2, ny // 2, ny)
        X, Y = np.meshgrid(x, y)  # Create 2D grid
        p_dict = cls.unpack_pars(psf_params)
        sigma = p_dict['FWHM'] / 2.355
        a = (np.cos(p_dict['theta'])**2)/(2*sigma**2) + (np.sin(p_dict['theta'])**2)/(2*sigma**2)
        b = -(np.sin(2*p_dict['theta']))/(4*sigma**2) + (np.sin(2*p_dict['theta']))/(4*sigma**2)
        c = (np.sin(p_dict['theta'])**2)/(2*sigma**2) + (np.cos(p_dict['theta'])**2)/(2*sigma**2)
        psf_image = p_dict['offset'] + p_dict['amplitude']*np.exp( - (a*((X-p_dict['xo'])**2) + 2*b*(X-p_dict['xo'])
                                                                      *(Y-p_dict['yo']) + c*((Y-p_dict['yo'])**2)))
        return ss.convolve2d(image, psf_image, mode='same')
    

class EMP_PSF(Jax_class):

    params = {'scale_factor': 1, 'offset': 1}

    def process_image(image, scale_factor=1, offset=1):
        scaled_image = (image[::scale_factor, ::scale_factor])[1::, 1::]
        cropped_image = image[70:210, 70:210]
        def safe_float32_conversion(value):
            try:
                return np.float32(value)
            except (ValueError, TypeError):
                print("This value is unjaxable: " + str(value))
        fin_image = np.nan_to_num(cropped_image)
        fin_image = np.vectorize(safe_float32_conversion)(fin_image)
        return fin_image

    img = process_image(fits.open("PSF/emp_psf.fits")[0].data[0,:,:])

    #define model function and pass independant variables x and y as a list
    @classmethod
    def generate(cls, image, psf_params):
        return ss.convolve2d(image, cls.img, mode='same')
    

class Winnie_PSF(Jax_class):

    @classmethod
    def init(cls, psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs):
        return WinniePSF(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs)

    @classmethod
    def pack_pars(cls, winnie_model):
        return winnie_model

    @classmethod
    def generate(cls, image, winnie_model):
        return np.mean(winnie_model.get_convolved_cube(image), axis=0)