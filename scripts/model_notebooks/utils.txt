# import libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

"""
Define utilization functions 
"""

def data_split(data, target, train_size=0.9, valid_size=0.2, scale=None):
    ''' A stratified data split into train, validation and test.
        uses target data for stratification.
    
    data :   nD array,
    target : 1d or 2d array,
    train_size : floating point number from 0 to 1,
             defines the size of training data compared to the whole data
    valid_size : floating point number from 0 to 1,
             defines the size of validation data compared to the training data
             in case when `valid_size` == 0.0 it performs just as skleran `train_test_split`
    scale :  apply scaler on the data, by default it is None
             valid inputs are 'StandardScaler' and 'MinMaxScaler'
    
    Note :   it returns tuple of 4 elements when `valid_size` == 0.0 and 6 otherwise
    '''
    # feature scalling by StandardScaler
    if scale == 'StandardScaler':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        scaled_bands = []
        for band in range(data.shape[3]):
            band_array = data[:, :, :, band]

            scaled_band = sc.fit_transform(band_array.reshape(
                band_array.shape[0], -1).T).T.reshape(band_array.shape)
            scaled_bands.append(scaled_band[:, :, :, np.newaxis])

        data = np.concatenate(scaled_bands, axis=-1)

    # feature scalling by MinMaxScaler
    elif scale == 'MinMaxScaler':
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()

        scaled_bands = []
        for band in range(data.shape[3]):
            band_array = data[:, :, :, band]

            scaled_band = sc.fit_transform(band_array.reshape(
                band_array.shape[0], -1).T).T.reshape(band_array.shape)
            scaled_bands.append(scaled_band[:, :, :, np.newaxis])

        data = np.concatenate(scaled_bands, axis=-1)

    elif scale == None:
        pass

    else:
        raise ValueError(
            "Wrong input for scaler. Should be 'StandardScaler', 'MinMaxScaler' or None ")


    if valid_size == 0.0:
        # split data to get the initial training test split
        X_train_cv, X_test, y_train_cv, y_test = train_test_split(data, target, random_state=1,
                                                                  train_size=train_size, stratify=target)
        out = X_train_cv, X_test, y_train_cv, y_test
        print(
            f'data split: \nTrain_CV:  {X_train_cv.shape[0]} \nTest: \t    {X_test.shape[0]}')

    else:

        # split data to get the initial training test split
        X_train_cv, X_test, y_train_cv, y_test = train_test_split(data, target, random_state=1,
                                                                  train_size=train_size, stratify=target)

        # split data to get train validation split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_cv, y_train_cv, random_state=1,
                                                              test_size=valid_size, stratify=y_train_cv)
        out = X_train, X_valid, X_test, y_train, y_valid, y_test
        print(
            f'data split: \nTrain: \t   {X_train.shape[0]} \nValidation: {X_valid.shape[0]} \nTest: \t    {X_test.shape[0]}')

    return out

def im_resize(img, dsize, interpolation):
    ''' Resize the image with given 2d sizes 
	and interpolation method: cv2.INTER_CUBIC #cv2.INTER_NEAREST'''
    resized = cv2.resize(img, dsize=dsize, interpolation=interpolation)
    return resized

def bbox_draw(im_array_label, im_array_vals=None, dsize=None):
    '''Draw rectangle box around the detected ship
    
    NOTE:  works if there is a single object detected on the image
           otherwhise it it draws a box including all the objects''' 
    
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

def model_history_plot(history, save=False):
    fig = plt.figure()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if save == True:
        fig.savefig('model_accuracy.png')

    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if save == True:
        fig.savefig('model_loss.png')

# Convert SAR backscattering DN to dB and vice bersa
natural2dB = lambda natural: np.log10(natural)*10
dB2natural = lambda dB: 10**(dB/10)

def lee_filter(img, size):
    '''
    Applies Lee Filter

    Parameters
    ----------
    img : array-like, image to apply filter on
    size : integer,
           The sizes of the uniform filter are given for each axis
    Returns
    -------
    Filtered array. Has the same shape as `input`.
    '''
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output
