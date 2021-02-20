import numpy as np

def detect_peaks(x, num_train, num_guard, rate_fa):
    """
    Detect peaks with CFAR algorithm.
    https://tsaith.github.io/detect-peaks-with-cfar-algorithm.html

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate. 
    """
    num_cells = x.size
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half

    alpha = num_train*(rate_fa**(-1/num_train) - 1)  # threshold factor

    peak_idx = []
    for i in range(num_side, num_cells - num_side):

        if i != i-num_side+np.argmax(x[i-num_side:i+num_side+1]):
            continue

        sum1 = np.sum(x[i-num_side:i+num_side+1])
        sum2 = np.sum(x[i-num_guard_half:i+num_guard_half+1])
        p_noise = (sum1 - sum2) / num_train
        threshold = alpha * p_noise

        if x[i] > threshold:
            peak_idx.append(i)

    peak_idx = np.array(peak_idx, dtype=int)

    return peak_idx


def CA_CFAR(rd_matrix, win_param, P_fa):
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
