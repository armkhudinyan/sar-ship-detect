"""
Define functions for data image reshape and visualization
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches


def im_resize(img, dsize, interpolation):
    ''' Resize the image with given 2d sizes 
	and interpolation method: cv2.INTER_CUBIC #cv2.INTER_NEAREST'''
    resized = cv2.resize(img, dsize=dsize, interpolation=interpolation)
    return resized

def bbox_draw(im_array_label, im_array_vals=None, dsize=None):
    '''Draw rectangle box around the detected ship''' 
    
	# define 2d image sizes
    if dsize is None:
        dsize = im_array_label.shape
    else:
        pass
	
    # resize the images to a given sizes
    y_pred_label_resized =  im_resize(im_array_label, dsize, interpolation=cv2.INTER_NEAREST)
    
    # get figure bounding parameters 
    indx = np.argwhere(y_pred_label_resized==1)
    up_b    = indx[:,0].min()
    down_b  = indx[:,0].max()
    left_b  = indx[:,1].min()
    right_b = indx[:,1].max()

    # set up the bbox corners coordinate with 1 pixel out
    upper_left = (left_b-0.75,up_b-0.75)
    width = right_b-left_b+1.5
    height = down_b-up_b+1.5
    
    if im_array_vals is not None:
        # resize the images to a given sizes
        y_pred_resized = im_resize(im_array_vals, dsize, interpolation=cv2.INTER_NEAREST)
        
        # Configure figure ploting
        plt.figure()

        #subplot(r,c) provide the no. of rows and columns
        f, ax = plt.subplots(1,2, figsize = (10,10)) 
        ax[0].imshow(y_pred_resized,cmap="gray")
        ax[0].set_title('bbox around the predicted ship')
        ax[1].imshow(y_pred_label_resized,cmap="gray")
        ax[1].set_title('bbox around the classified ship')

        # Create a Rectangle patch
        #Rectangle((left,upper),width,height))
        rect = patches.Rectangle(
            upper_left,
            width,height,
            linewidth=0.7,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax[0].add_patch(rect)
        #ax[1].add_patch(rect)
        plt.show()
    else:
        # Configure figure ploting
        plt.figure()

        #subplot(r,c) provide the no. of rows and columns
        f, ax = plt.subplots(1, figsize=(5,5)) 
        ax.imshow(y_pred_label_resized,cmap="gray")
        ax.set_title('bbox around the predicted ship')

        # Create a Rectangle patch
        #Rectangle((left,upper),width,height))
        rect = patches.Rectangle(
            upper_left,
            width,height,
            linewidth=0.7,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()