import numpy as np

NTypes = np.ceil(1003 * np.asarray([47, 1.55, 1.55, 1.3, 1.3, 2.6, 2.1,
                                      38, 0.25, 0.25, 0.25, 0.25, 1.8, 1.8]) / 100)

print(sum(NTypes))