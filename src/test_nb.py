## Our goal is to plot graph of sin(x) on discrete set
##
## (Michal Barnisin, Simon Varga)

## Lets start with importing neccesarry packages:

import numpy as np
import matplotlib.pyplot as plt

## Now, we create X points 
xs = np.arange(-10, 11, 0.1)
## ... and corresponding Y points
ys = np.ndarray(xs.shape)

for i in range(xs.size):
    ys[i] = np.sin(xs[i])
# Should be done as np.sin(xs), but we want to test indenting
# And now also commentars, not markdown cells

    print(xs[i], end='')
    print(': ', end='')
    print(ys[i])
## And now plot the function!
plt.scatter(xs, ys)
plt.show()

## That's it :)

print('Have a nice day')
