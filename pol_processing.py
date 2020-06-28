'''
A module for doing polarisation photography.
Would be nice to wrap a GUI around it at some point

Written by Scott Silburn.
'''
import rawpy
import copy
import numpy as np
import cv2


def cvtcode(raw,dst='RGB'):
    '''
    Get the correct OpenCV code to give to cvtColor() when debayering a raw image.

    Parameters:

        raw (RawPy) : RawPy instance for the RAW file you want to debayer
        dst (str)   : Target colour space for debayering, default is RGB.

    Returns:
        int : OpenCV cvtColor code.
    '''
    ch0 = raw.raw_colors_visible[1,1]
    ch1 = raw.raw_colors_visible[1,2]
    bayer_type = raw0.color_desc.decode('utf-8')[ch0] + raw0.color_desc.decode('utf-8')[ch1]
    cvtcode = getattr(cv2, 'COLOR_BAYER_{:s}2{:s}'.format(bayer_type,dst.upper()))

    return cvtcode


def gamma_8bit(im,gamma=2.2):
    '''
    Take a linear image and convert it to an 8-bit per pixel image
    with correction to a given gamma

    Parameters:
        im (array) : Input linear image
        gamma (float) : Desired gamma

    Returns:

        uint8 array : 8-bit gamma corrected image
    '''
    return (255 * (im / im.max())**(1 /gamma)).astype(np.uint8)


def process_polarisation(raw0,raw45,raw90,gamma=2.2):
    '''
    Given 3 raw file objects for images with taken a polarising filter at 0, 45 and 90 degrees,
    separate the linearly polarised and not-linearly polarised images, calculate the degree of linear
    polarisation and linear polarisation angle.

    The first 3 returned images are de-bayered (i.e. RGB colour), gamma corrected according to
    the gamma input argument and white balanced according to the in-camera white balance setting.

    Parameters:

        raw0 (PyRaw)  : RawPy instance for the raw file with polariser at 0 degrees.
        raw45 (PyRaw) : RawPy instance for the raw file with polariser at 45 degrees.
        raw90 (PyRaw) : RawPy instance for the raw file with polariser at 90 degrees.
        gamma (float) : Gamma value for RAW conversion

    Returns:

        h*w*3 array  : 8-bit per pixel RGB image including all light.
        h*w*3 array  : 8-bit per pixel RGB image including only light which is not linearly polarised light.
        h*w*3 array  : 8-bit per pixel RGB image including only linearly polarised light.

         h*w*3 array :  Float array the same size as the images containing the degree of linear polarisation for each pixel
                        and colour channel. Values range from 0 (no linear polarisation) to 1 (completely linearly polarised).

         h*w*3 array : Float array the same size as the images containing the angle of the linear polarisation \
                       in degrees for each pixel and colour channel. Measured anti-clockwise with 0 degrees meaning \
                       aligned with polariser axis in the raw0 image.

    '''

    # Get raw linear sensor data from RAW files
    im0 = copy.deepcopy(raw0.raw_image_visible).astype(np.float32)
    im45 = copy.deepcopy(raw45.raw_image_visible).astype(np.float32)
    im90 = copy.deepcopy(raw90.raw_image_visible).astype(np.float32)

    # Get the "as shot" white balance from the RAW file and apply to the images
    wb_coeffs0 = np.array(raw0.camera_whitebalance)
    wb_coeffs0 = wb_coeffs0 / wb_coeffs0.max()
    wb_coeffs45 = np.array(raw45.camera_whitebalance)
    wb_coeffs45 = wb_coeffs45 / wb_coeffs45.max()
    wb_coeffs90 = np.array(raw90.camera_whitebalance)
    wb_coeffs90 = wb_coeffs90 / wb_coeffs90.max()

    # Black level and white balance corrections
    for channel in range(4):
        im0[raw0.raw_colors_visible == channel] = (im0[raw0.raw_colors_visible == channel] - raw0.black_level_per_channel[channel]) * wb_coeffs0[channel]
        im45[raw45.raw_colors_visible == channel] = (im45[raw45.raw_colors_visible == channel] - raw45.black_level_per_channel[channel]) * wb_coeffs45[channel]
        im90[raw90.raw_colors_visible == channel] = (im90[raw90.raw_colors_visible == channel] - raw90.black_level_per_channel[channel]) * wb_coeffs90[channel]


    # Calculate polarised and un-polarised brightness and polarisation angle at each pixel
    im_polarised = np.sqrt((im0 - im90)**2 + (2*im45 - im0 - im90)**2)
    im_unpolarised = im0 + im90 - im_polarised
    theta = 90 - 180 * np.arctan2(2 * im45 - im0 - im90, im90 - im0) / (2 * np.pi)

    # Get rid of any negative values due to dark subtraction and calculations then convert back
    im_polarised[im_polarised < 0] = 0
    im_unpolarised[im_unpolarised < 0] = 0

    # Calculate degree of linear polarisation
    dolp = im_polarised / (im_unpolarised + im_polarised)


    # Convert calculated images to original data format
    im_polarised = im_polarised.astype(raw0.raw_image_visible.dtype)
    im_unpolarised = im_unpolarised.astype(raw0.raw_image_visible.dtype)


    # Debayer the images
    im_all = cv2.cvtColor(im_unpolarised + im_polarised,cvtcode(raw0))
    im_polarised = cv2.cvtColor(im_polarised,cvtcode(raw0))
    im_unpolarised = cv2.cvtColor(im_unpolarised,cvtcode(raw0))

    # Debayer the degree of polarisation
    dolp = cv2.cvtColor( (dolp * (2**16-1)).astype(np.uint16),cvtcode(raw0))
    dolp = dolp.astype(np.float16) / (2**16-1)


    # The polarisation angle doesn't behave well with normal debayering
    # because the denegeracy in 180 degrees makes averaging go wrong.
    # So instead we have to calculate a half-resolution theta image
    # and scale it up.
    theta_debayered = np.zeros((theta.shape[0]//2,theta.shape[1]//2,3),dtype=np.float16)
    for ci in range(2):
        for cj in range(2):
            channel = ['R','G','B'].index(raw0.color_desc.decode('utf-8')[raw0.raw_colors_visible[ci,cj]])
            theta_debayered[:,:,channel] = theta[ci::2,cj::2]

    theta_debayered = ((theta_debayered / 180) * (2**16-1)).astype(np.uint16)
    theta_debayered = cv2.resize(theta_debayered,(theta.shape[1],theta.shape[0]),interpolation=cv2.INTER_NEAREST)
    theta_debayered = 180 * theta_debayered.astype(np.float32) / (2**16-1)

    return gamma_8bit(im_all,gamma=gamma),gamma_8bit(im_unpolarised,gamma=gamma),gamma_8bit(im_polarised,gamma=gamma),dolp,theta_debayered



def make_falsecolour(im=None,dolp=None,theta=None,channel=None):
    '''
    Create a false-colour image where the brightness shows the image brightness,
    saturation shows the degree of linear polarisation, and hue shows the angle of linear
    polarisation.

    Parameters:

        im            : h*w*3 RGB image to use for brightness information
        dolp          : h*w*3 array with degree of linear polarisation
        theta         : h*w*3 array with linear polarisation angle in degrees
        channel (str) : Specified which colour channel to use data from, 'R', 'G', 'B' or None.\
                        If None, the average data from all 3 channels is ued.
    Returns:

        h*w*3 array containing 8-bit per pixel RGB false colour image.
    '''

    if im is None and dolp is None and theta is None:
        raise ValueError('You must provide at least one of im, dolp, theta!')


    if im is None:
        if dolp is not None:
            im = (255 *dolp).astype(np.uint8)
        elif theta is not None:
            im = 255 * np.ones(theta.shape,dtype=np.uint8)

    if dolp is None:
        if im is not None:
            dolp =  np.ones(im.shape,dtype=np.uint8)
        elif theta is not None:
            im = 255 * np.ones(theta.shape,dtype=np.uint8)

    if theta is None:
        dolp = np.zeros(im.shape,dtype=np.uint8)
        theta = dolp

    if channel is None:
        # Get  brightness image from value in HSV colour space
        i0 = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 2]

        dolp = dolp.mean(axis=2)

        # We can't just mean the angles because the 180 degree degeneracy messes it up.
        # So we have to check and do some tweaking before average.
        thetamin = theta.min(axis=2)
        thetamax = theta.max(axis=2)
        thetamean = theta.mean(axis=2)

        theta_fixed = copy.copy(theta)
        for i,j in np.argwhere(thetamax - thetamin > 90):
            ind = np.argmax(np.abs(theta_fixed[i,j,:] - thetamean[i,j]))
            theta_fixed[i,j,ind] = 180 - theta_fixed[i,j,ind]


        theta = theta_fixed.mean(axis=2)

    else:
        channel = ['R','G','B'].index(channel)

        i0 = im[:,:,channel]
        theta = theta[:,:,channel]
        dolp = dolp[:,:,channel]

    h = theta.astype(np.uint8)
    s = (dolp * 255).astype(np.uint8)
    v = i0

    hsvim = np.dstack( (h[:,:,np.newaxis],s[:,:,np.newaxis],v[:,:,np.newaxis]))
    rgbim = cv2.cvtColor(hsvim,cv2.COLOR_HSV2RGB)

    return rgbim



if __name__ == '__main__':
    '''
    A demonstration of using all of this: take 3 raw files and make some fancy images.
    '''

    import matplotlib.pyplot as plt

    # Open the camera RAW files
    raw0 = rawpy.imread('desk0.arw')
    raw45 = rawpy.imread('desk45.arw')
    raw90 = rawpy.imread('desk90.arw')

    # Process them to get the polarisation information
    im_all,im_unpol,im_pol,dolp,theta = process_polarisation(raw0,raw45,raw90)

    # Close the RAW files
    raw0.close()
    raw45.close()
    raw90.close()


    # Write our polarisation images to output files.
    # Note OpenCV's image writing takes images in BGR channel order.
    cv2.imwrite('polim_all.jpg',cv2.cvtColor(im_all,cv2.COLOR_RGB2BGR))
    cv2.imwrite('polim_unpolarised.jpg', cv2.cvtColor(im_unpol, cv2.COLOR_RGB2BGR))
    cv2.imwrite('polim_polarised.jpg', cv2.cvtColor(im_pol, cv2.COLOR_RGB2BGR))

    # Make a nice false colour image and also save that
    fc = make_falsecolour(im=im_all,dolp=dolp,theta=theta)
    cv2.imwrite('polim_falsecol_all.jpg', cv2.cvtColor(fc, cv2.COLOR_RGB2BGR))

    # Make another false colour image showing just the degree of linear polarisation
    fc = make_falsecolour(dolp=dolp)
    cv2.imwrite('polim_falsecol_dolp.jpg', cv2.cvtColor(fc, cv2.COLOR_RGB2BGR))

    # Make another false colour image just showing the polarisation angle
    fc = make_falsecolour(theta=theta)
    cv2.imwrite('polim_falsecol_theta.jpg', cv2.cvtColor(fc, cv2.COLOR_RGB2BGR))


