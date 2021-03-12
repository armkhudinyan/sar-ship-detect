import sys
import numpy as np
import time

def SAR_ObjectDetect_CNN(nrcs, inc_angle, cnn_model, window=(80, 80)):
    '''Detect objects on the SAR image
    
    nrcs : Normalized Radar Cross Section [linear units]
    inc_angle : incidence angle
    cnn_model: CNN model (to be loaded before inputing)
    window: window size to be used as CNN input data shape
    
    Returns 
    -------
            Classified image with predicted objects 
            
    NOTE:   All inputs must be Numpy arrays of equal sizes
    '''

    t0 = time.time()

    image = nrcs.copy()
    angle = inc_angle.copy()

    y, x = window
    # symply for tracking the progress
    num_iter = (image.shape[0]//x)*(image.shape[1]//y)
    # remove image edges wthat don't fit in the filter
    image = image[:int(image.shape[0]//y)*y, :int(image.shape[1]//x)*x]
    angle = angle[:int(angle.shape[0]//y)*y, :int(angle.shape[1]//x)*x]

    # list of outputs from each window
    out_pred = []

    i_num = []
    iteration = 0
    for i in range(0, image.shape[0], y):
        j_num = []
        for j in range(0, image.shape[1], x):
            # get the window to apply the complete chain of analysis
            sub_image_VV = image[i:i+y, j:j+x]
            sub_image_angle = angle[i:i+y, j:j+x]

            test_data = np.concatenate(
                [sub_image_VV[np.newaxis, :, :, np.newaxis], sub_image_angle[np.newaxis, :, :, np.newaxis]], axis=-1)

            #============================
            # Predict objects
            #============================
            y_pred_class = cnn_model.predict_classes(
                test_data, batch_size=1, verbose=0)  # batch_size=64, verbose=1
            out_pred.append(y_pred_class)

            j_num.append(j)

            iteration += 1
            sys.stdout.write(f"\rprocessingn window N: {iteration}/{num_iter}")
            sys.stdout.flush()

        i_num.append(i)

    # reshape back into image
    width = image.shape[1]//y
    sar_classified = np.vstack([np.hstack(out_pred[i:i+width])
                                for i in range(0, len(out_pred), width)])

    t1 = time.time()
    run_time = round((t1-t0)/60, 2)
    print(f'\nruntime: {run_time} mins')

    return sar_classified
