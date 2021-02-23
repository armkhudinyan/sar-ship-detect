import numpy as np
from scipy import signal

def CA_CFAR_naive(rd_matrix, win_param, P_fa):
    '''
    Cell Averaging - Constant False Alarm Rate algorithm

    Parameters:
    -----------
    rd_matrix : range-Doppler matrix (sigma or gamma naught data)
    win_param : Parameters of the noise power estimation window
                    [Est. window length, Est. window width, Guard window length, Guard window width]
    P_fa : Probability of a false-alarm 
    '''
    # -- Set inital parameters --
    win_len = win_param[0]
    win_width = win_param[1]
    guard_len = win_param[2]
    guard_width = win_param[3]

    norc = np.size(rd_matrix, 1)  # number of range cells
    noDc = np.size(rd_matrix, 0)  # number of Doppler cells
    hit_matrix = np.zeros((noDc, norc), dtype=float)

    # Generate window mask
    rd_block = np.zeros((2 * win_width + 1, 2 * win_len + 1), dtype=float)
    mask = np.ones((2 * win_width + 1, 2 * win_len + 1))
    mask[win_width - guard_width:win_width + 1 + guard_width, win_len - guard_len:win_len + 1 + guard_len] = np.zeros(
        (guard_width * 2 + 1, guard_len * 2 + 1))

    # number of training cells
    num_train = np.sum(mask)
    # calculated threshold factor
    alpha = num_train*(P_fa**(-1/num_train) - 1)  # threshold factor
    print(alpha)
    # -- Perform automatic detection --
    for j in np.arange(win_width, noDc - win_width, 1):  # Range loop
        for i in np.arange(win_len, norc - win_len, 1):  # Doppler loop
            rd_block = rd_matrix[j - win_width:j +
                                 win_width + 1, i - win_len:i + win_len + 1]
            rd_block = np.multiply(rd_block, mask)
            # Threshold level above the estimated average noise power
            Threshold = alpha * (np.sum(rd_block)/num_train)

            if rd_matrix[j, i] > Threshold:
                hit_matrix[j, i] = 1

    return hit_matrix

def CA_CFAR(rd_matrix, win_param, P_fa=10e-4):
    """
    Description:
    ------------
        Cell Averaging - Constant False Alarm Rate algorithm
        Performs an automatic detection on the input range-Doppler matrix with an adaptive thresholding.

        Implem. based on https://www.jku.at/fileadmin/gruppen/183/Docs/Finished_Theses/Bachelor_Thesis_Katzlberger_final.pdf
    ---------------------
    Parameters:
    -----------
    rd_matrix : range-Doppler matrix (sigma or gamma naught data)
    win_param : Parameters of the noise power estimation window
                    [Est. window length, Est. window width, Guard window length, Guard window width]
    P_fa : Probability of a false-alarm 
    
    Returns:
    --------
           Calculated hit matrix
    """
    # Set inital parameters
    win_width = win_param[0]
    win_height = win_param[1]
    guard_width = win_param[2]
    guard_height = win_param[3]

    # Create window mask with guard cells
    mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
    mask[win_height - guard_height:win_height + 1 + guard_height,
         win_width - guard_width:win_width + 1 + guard_width] = 0

    # Number cells within window around CUT; used for averaging operation.
    num_valid_cells_in_window = signal.convolve2d(
        np.ones(rd_matrix.shape, dtype=float), mask, mode='same')

    # Calculate scaling factor of threshold with false alarm probability (P_fa)
    def alpha(num_train): return num_train*(P_fa**(-1/num_train) - 1)
    # scaling factor of threshold (aka alpha)
    scaling_factor_in_window = alpha(num_valid_cells_in_window)

    # Perform detection
    rd_windowed_sum = signal.convolve2d(rd_matrix, mask, mode='same')
    rd_avg_noise_power = rd_windowed_sum / num_valid_cells_in_window
    threshold = scaling_factor_in_window * rd_avg_noise_power  # threshold value
    hit_matrix = rd_matrix > threshold

    return hit_matrix
