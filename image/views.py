import os
os.environ["WEBBPSF_PATH"] = 'webbpsf-data'
os.environ["WEBBPSF_EXT_PATH"] = 'webbpsf-data'
os.environ["PYSYN_CDBS"] = "cdbs"

import numpy as np
from PIL import Image
from io import BytesIO
from django.http import HttpResponse
from rest_framework.decorators import api_view
import json
from matplotlib import cm
import jax.numpy as jnp
from .objective_functions import Parameter_Index, objective_model
from .SLD_utils import DoubleHenyeyGreenstein_SPF, DustEllipticalDistribution2PowerLaws, Winnie_PSF
from .new_SLD_utils import EMP_PSF
from .SLD_ojax import ScatteredLightDisk

x1 = 0
x2 = 4096
y1 = 0
y2 = 4096
psf_dir = "for_mihir_20241218/F300M/"
psf_suffixes = "_F300M.npy"

test_parangs = jnp.array([0, 90, 180, 270])

#Read in the pre-generated PSFs
im_mask_rolls = jnp.load(psf_dir+"im_mask_rolls"+psf_suffixes)
psf_inds_rolls = jnp.load(psf_dir+"psf_inds_rolls"+psf_suffixes)
psf_offsets = jnp.load(psf_dir+"psf_offsets"+psf_suffixes)
psfs = jnp.load(psf_dir+"psfs"+psf_suffixes)

unique_inds = jnp.unique(psf_inds_rolls)
n_unique_inds = len(unique_inds)

winnie_300FM = Winnie_PSF.init(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, test_parangs, n_unique_inds)

psf_dir = "for_mihir_20241218/F360M/"
psf_suffixes = "_F360M.npy"

#Read in the pre-generated PSFs
im_mask_rolls = jnp.load(psf_dir+"im_mask_rolls"+psf_suffixes)
psf_inds_rolls = jnp.load(psf_dir+"psf_inds_rolls"+psf_suffixes)
psf_offsets = jnp.load(psf_dir+"psf_offsets"+psf_suffixes)
psfs = jnp.load(psf_dir+"psfs"+psf_suffixes)

unique_inds = jnp.unique(psf_inds_rolls)
n_unique_inds = len(unique_inds)

winnie_360FM = Winnie_PSF.init(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, test_parangs, n_unique_inds)


def get_image(alpha_in, alpha_out, sma, e, inclination, position_angle, x_center, y_center, g1, g2, weight, psf,
              parang1, parang2, parang3, parang4):

    spf_params = DoubleHenyeyGreenstein_SPF.params
    misc_params = Parameter_Index.misc_params

    test_parangs = jnp.array([parang1, parang2, parang3, parang4])
    
    psf_params = None
    
    if psf == "NIRCAM 300FM":
        psf_params = winnie_300FM
        winnie_300FM.psf_parangs = test_parangs
    elif psf == "NIRCAM 360FM":
        psf_params = winnie_360FM
        winnie_360FM.psf_parangs = test_parangs

    spf_params['g1'] = g1
    spf_params['g2'] = g2
    spf_params['weight'] = weight

    disk_params = Parameter_Index.disk_params
    disk_params['alpha_in'] = alpha_in
    disk_params['alpha_out'] = alpha_out
    disk_params['sma'] = sma
    disk_params['e'] = e
    disk_params['inclination'] = inclination
    disk_params['position_angle'] = position_angle
    disk_params['x_center'] = x_center
    disk_params['y_center'] = y_center

    misc_params['nx'] = 500
    misc_params['ny'] = 500

    if psf == "EMPIRICAL":
        img = objective_model(disk_params, spf_params, EMP_PSF.params, misc_params,
                                ScatteredLightDisk, DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, EMP_PSF)
    elif psf_params != None:
        img = objective_model(disk_params, spf_params, psf_params, misc_params,
                                ScatteredLightDisk, DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, Winnie_PSF)
    else:
        img = objective_model(disk_params, spf_params, psf_params, misc_params,
                                ScatteredLightDisk, DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, None)
    
    return img

@api_view(['POST'])
def generate_image(request):
    try:
        # Parse JSON data from request body
        data = json.loads(request.body.decode('utf-8'))
        
        # Extract parameters
        alpha_in = float(data.get("alpha_in", 5))
        alpha_out = float(data.get("alpha_out", -5))
        sma = float(data.get("sma", 50))
        e = float(data.get("e", 0.0))
        inclination = float(data.get("inclination", 0))
        position_angle = float(data.get("position_angle", 0))
        x_center = float(data.get("x_center", 70.0))
        y_center = float(data.get("y_center", 70.0))
        g1 = float(data.get("g1", 0.5))
        g2 = float(data.get("g2", 0.5))
        weight = float(data.get("weight", 0.5))
        psf = data.get("psf", "NONE")
        parang1 = float(data.get("parang1", 0.0))
        parang2 = float(data.get("parang2", 90.0))
        parang3 = float(data.get("parang3", 180.0))
        parang4 = float(data.get("parang4", 270.0))

        # For now, generate a dummy black image (you can modify this logic)
        image_array = np.asanyarray(get_image(alpha_in, alpha_out, sma, e, inclination, position_angle, x_center, y_center, g1, g2,
                                              weight, psf, parang1, parang2, parang3, parang4))
        
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
            image_array = (255 * (image_array - image_array.min()) / (image_array.max() - image_array.min())).astype(np.uint8)

        normalized_img = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        inferno_colormap = cm.inferno(normalized_img)

        # ✅ Convert to 8-bit RGB (Matplotlib returns floats, so scale to 255)
        inferno_rgb = (inferno_colormap[:, :, :3] * 255).astype(np.uint8)
        
        # Convert NumPy array to PIL Image
        img = Image.fromarray(inferno_rgb, mode="RGB")

        # Save image to a buffer
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return HttpResponse(buffer, content_type="image/png")

    except Exception as e:
        print(e)
        return HttpResponse(f"Error processing request", status=400)
        # return Image.fromarray(np.zeros((140, 140, 3), dtype=np.uint8))
