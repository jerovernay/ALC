import numpy as np
import matplotlib.pyplot as plt

#3

1 + 3

a = 7

b = a +1

#print("b = " , b)

v = np.array([ 1, 2 , 3 , -1 ])

w = np.array( [2 , 3 , 0 , 5 ])

# print( "v + w =  ", v + w)
# print( " 2 ∗ v = ", 2 * v )
# print( "v ∗∗2 = " , v ** 2)

A = np.array( [ [ 1 , 2 , 3 , 4 , 5 ] , [ 0 , 1 , 2 , 3 , 4 ] , [ 2 , 3 , 4 , 5 , 6 ] , [ 0 , 0 , 1 , 2 , 3 ] , [ 0 , 0 , 0 , 0 , 1 ] ] )
#print(A )

ind = np.array( [ 0 , 2 , 4 ] )
#print(ind)

#4
a = -(3/2)
b = 11/2
c = -3

xx = np.array([1,2,3])
yy = np.array([1,2,0])
x = np.linspace(0, 4, 100)
f = lambda t : a*t**2 + b * t + c
plt.plot(xx, yy, '*')
plt.plot(x, f(x))
plt.show() 