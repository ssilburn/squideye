# SquidEye - Polarisation Photography

This is a small tool for doing Polarisation photography, inspired by u/OmniaMors on Reddit and related stuff I've done at work before. This is a quick first commit - much more development of both the code and documentation will happen if I have time.

## What is it for?
This is a small Python code for taking photographs taken with polarising filters, with the polariser rotated at different angles, and calculating from those the properties of the polarised light. This can then be used to make cool images which allow us to see things we usually can't.

For example, it allows you to take photos where you can separate out polarised and un-polarised light, like this photo of my desk:

![enter image description here](http://internalreflections.co.uk/imghost/pol_slideshow.gif)

You could also make an image which uses the image brightness to show the normal brightness, the saturation to show the degree of polarisation, and the colour for the angle of polarisation, like this:

![enter image description here](http://www.internalreflections.co.uk/imghost/polim_falsecol_all.jpg)(I like how this makes IMAX 3D glasses look like the old red/blue kind!)

Or we can make an image which is dark where the light is not polarised, and light where it is polarised:
![enter image description here](http://www.internalreflections.co.uk/imghost/polim_falsecol_dolp.jpg)

Of if we want to get very colourful, make an image which has bright colours everywhere to show the angle of all the polarised light in the scene:

![enter image description here](http://www.internalreflections.co.uk/imghost/polim_falsecol_theta.jpg)
## How do I use it?
### What images do I need to take?
You'll need a camera that can shoot RAW, and a polarising filter. You also need to pick a subject which is still enough that you can take 3 photos of it, and rotate a filter in between, without your subject moving. Also the lighting should not change between the exposures.

You need to take 3 photos of your subject, with the exposure (aperture, shutter speed & ISO)  and white balance on the camera locked to be the same for all 3 photos. For these 3 photos you need to have a polarising filter on the camera. You then take 3 photos:

1) First photo with the polariser at an angle of your choice.
2. Second photo with the polariser rotated 45 degrees relative to the first photo. It's important that this 45 degree rotation is pretty accurate
3. Third photo with the polariser rotated 90 degrees relative to photo 1, i.e. rotated another 45 degrees from photo 2.

Then you need the RAW files from those photos on your computer.

### How do I use the code?
At some point I'd like to write a GUI for this so it can be used without having to know Python / programmatic image manipulation, but for now you have to edit the provided python script to use the code.

To use the script you will need Python and the following python modules, all readily available on PyPI:
rawpy
opencv-python (a.k.a. cv2)
numpy

If you open up the pol_processing.py file and scroll to the bottom, you'll see these lines:

    raw0 = rawpy.imread('desk0.arw')  
	raw45 = rawpy.imread('desk45.arw')  
	raw90 = rawpy.imread('desk90.arw')

Change the part in brackets and single quotes to be the location of your RAW files for the 0, 45 and 90 degree polariser images (these are relative paths to where the python file is). You should then be able to run `Python pol_processing.py` and It should then run and produce some fun output images. If you're of a more technical disposition, have a look around the code and do some more fun stuff!
