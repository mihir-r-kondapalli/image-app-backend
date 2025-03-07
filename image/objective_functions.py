import jax
import jax.numpy as jnp
from functools import partial

class Parameter_Index:
    
    disk_params = {'accuracy': 5.e-3, 'alpha_in': 5, 'alpha_out': -5, 'sma': 50, 'e': 0., 'ksi0': 3., 'gamma': 2., 'beta': 1., 'amin': 0.,
                'dens_at_r0': 1., 'inclination': 0, 'position_angle': 0, 'x_center': 70., 'y_center': 70., 'halfNbSlices': 25, 'omega': 0.,}

    misc_params = {'distance': 50., 'pxInArcsec': 0.01414, 'nx': 140, 'ny': 140, 'halfNbSlices': 25, 'flux_scaling': 1e6}  # Don't change this

    #####
    # Parameter Dictionaries for SPFs and PSFs are given in SLD_utils.py in their param fields
    # params for InterpolatedUnivariateSpline_SPF and Winne_PSF is just an instance of WinniePSF (winnie_class.py) itself


def pack_pars(p_dict, orig_dict):
    """
    This function takes a parameter dictionary and packs it into a JAX array
    where the order is set by the parameter name list defined on the class.
    """    
    p_arrs = []
    for name in orig_dict.keys():
        p_arrs.append(p_dict[name])
    return jnp.asarray(p_arrs)


@partial(jax.jit, static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'PSFModel', 'nx', 'ny', 'halfNbSlices'])
def jax_model(DiskModel, DistrModel, FuncModel, PSFModel, disk_params, spf_params, psf_params, distance = 0., pxInArcsec = 0.,
              nx = 140, ny = 140, halfNbSlices = 25, flux_scaling = 1e6):

    distr_params = DistrModel.init(accuracy=disk_params[0], ain=disk_params[1], aout=disk_params[2], a=disk_params[3],
                                   e=disk_params[4], ksi0=disk_params[5], gamma=disk_params[6], beta=disk_params[7],
                                   amin=disk_params[8], dens_at_r0=disk_params[9])
    disk_params_jax = DiskModel.init(distr_params, disk_params[10], disk_params[11],
                                              disk_params[1], disk_params[2], disk_params[3],
                                              nx=nx, ny=ny, distance = distance,
                                              omega = disk_params[15], pxInArcsec=pxInArcsec)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    scattered_light_image = DiskModel.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, spf_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)
    
    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - disk_params[12] + xc
    y = y - disk_params[13] + yc
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    if PSFModel != None:
        scattered_light_image = PSFModel.generate(scattered_light_image, psf_params)

    return scattered_light_image*flux_scaling

def objective_model(disk_params, spf_params, psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, **kwargs):
    """
    Objective function for optimization that updates only the selected parameters.
    """

    model_image = jax_model(
        DiskModel, DistrModel, FuncModel, PSFModel,
        pack_pars(disk_params, Parameter_Index.disk_params) if isinstance(disk_params, dict) else disk_params,
        FuncModel.pack_pars(spf_params) if isinstance(spf_params, dict) else spf_params,
        PSFModel.pack_pars(psf_params) if isinstance(psf_params, dict) else psf_params,
        distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
        nx = misc_params['nx'], ny = misc_params['ny'],
    )

    return model_image


def objective_ll(disk_params, spf_params, psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, target_image, err_map, **kwargs):
    """
    Objective function for optimization that updates only the selected parameters.
    """

    model_image = objective_model(
        disk_params, spf_params, psf_params, misc_params, DiskModel, DistrModel, FuncModel, PSFModel
    )

    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)

    return -0.5 * jnp.sum(result)  # / jnp.size(target_image)

@jax.jit
def log_likelihood(image, target_image, err_map):
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)

    return -0.5 * jnp.sum(result)  # / jnp.size(target_image)


def objective_fit(params_fit, fit_keys, disk_params, spf_params, psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, target_image, err_map, **kwargs):
    """
    Objective function for optimization that updates only the selected parameters.
    """

    # These temporary dictionaries are edited based on params_fit
    temp_disk_params = disk_params.copy() if isinstance(disk_params, dict) else {}
    temp_spf_params = spf_params.copy() if isinstance(spf_params, dict) else {}
    temp_psf_params = psf_params.copy() if isinstance(psf_params, dict) else {}
    temp_misc_params = misc_params.copy() if isinstance(misc_params, dict) else {}

    # Corresponding index of params_fit for each key in fit_keys
    param_index = 0

    for key in fit_keys:
        if key in temp_disk_params:
            temp_disk_params[key] = params_fit[param_index]
        elif key in temp_spf_params:
            temp_spf_params[key] = params_fit[param_index]
        elif key in temp_psf_params:
            temp_psf_params[key] = params_fit[param_index]
        elif key in temp_misc_params:
            temp_misc_params[key] = params_fit[param_index]
        param_index += 1


    if PSFModel != None:
        model_image = jax_model(
            DiskModel, DistrModel, FuncModel, PSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(temp_spf_params) if isinstance(spf_params, dict) else spf_params,
            PSFModel.pack_pars(temp_psf_params) if isinstance(psf_params, dict) else psf_params,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    else:
        model_image = jax_model(
            DiskModel, DistrModel, FuncModel, PSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(temp_spf_params) if isinstance(spf_params, dict) else spf_params,
            None,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )

    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)

    return -0.5 * jnp.sum(result)  # / jnp.size(target_image)