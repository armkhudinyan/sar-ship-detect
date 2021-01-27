# import libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.model_selection import train_test_split

"""
Define utilization functions 
"""

def data_split(data, target, train_size=0.9, valid_size=0.2, scale=False):
    ''' A stratified data split into train, validation and test.
        uses target data for stratification.
    
    data :   nD array,
    target : 1d or 2d array,
    train_size : floating point number from 0 to 1,
             defines the size of training data compared to the whole data
    valid_size : floating point number from 0 to 1,
             defines the size of validation data compared to the training data
    scale :  apply scaler on the data,
             the scaler id from sklearn.preprocessing.StandardScaler'''

    # feature scalling
    if scale == True:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        scaled_bands = []
        for band in range(data.shape[3]):
            band_array = data[:, :, :, band]

            scaled_band = sc.fit_transform(
                band_array.reshape(
                band_array.shape[0], -1).T
                ).T.reshape(band_array.shape)
            scaled_bands.append(scaled_band[:, :, :, np.newaxis])

        data = np.concatenate(scaled_bands, axis=-1)

    # split data to get the initial training test split
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=1,
                                                        train_size=train_size, stratify=target)

    # split data to get train validation split
    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, random_state=1,
                                                                test_size=valid_size, stratify=y_train)

    print(
        f'data split: \nTrain: \t   {X_train_cv.shape[0]} \nValidation: {X_valid.shape[0]} \nTest: \t    {X_test.shape[0]}')

    return X_train_cv, X_valid, X_test, y_train_cv, y_valid, y_test

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
