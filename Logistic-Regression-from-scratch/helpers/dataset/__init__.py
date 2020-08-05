import numpy as np
from tensorflow.keras.datasets import mnist

def get_data():
    """
    Returns (x_train, y_train), (x_test, y_test)
    Only examples that are either hand written 0s or 1s are kept
    All image examples are normalized (divided by 255.)
    """
    def extract_zeros_and_ones(x, y):
        i_zeros = np.where(y == 0)
        i_ones = np.where(y == 1)
        x_zeros = x[i_zeros]
        x_ones = x[i_ones]
        x = np.concatenate((x_zeros, x_ones), axis=0)
        y = np.array([0] * x_zeros.shape[0] + [1] * x_ones.shape[0])
        return x/255., y
    print('Loading data..')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = extract_zeros_and_ones(x_train, y_train)
    x_test, y_test = extract_zeros_and_ones(x_test, y_test)
    print('Done.')
    return (x_train, y_train), (x_test, y_test)

def get_random_batch(x, y, batch_size):
    """
    Returns examples, labels randomly selected
    from given x and y
    """
    num_features = x.shape[1] * x.shape[1]
    num_total = x.shape[0]
    # Examples are unrolled
    X = np.zeros((batch_size, num_features))
    Y = np.zeros((batch_size, 1))

    indices = np.random.randint(0, num_total, batch_size)
    
    for i, index in enumerate(indices):
        X[i] = np.reshape(x[index], (num_features,))
        Y[i] = np.array(y[index])

    return X, Y