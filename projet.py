import numpy as np
import matplotlib.pyplot as plt

### Question 4

def grad(u):
    """Returns the discrete gradient of matrix u."""
    (n,m,p) = u.shape # 3rd component is for RGB
    grad_x, grad_y = np.zeros((n,m,p)), np.zeros((n,m,p))
    
    #grad_x
    for i in range(n-1): # i < n
        for j in range (m):
            grad_x[i,j] = u[i+1,j]-u[i,j]
    # The rest are already zeros.

    #grad_y
    for j in range(m-1): # i < n
        for i in range (n):
            grad_y[i,j] = u[i,j+1]-u[i,j]

    return(grad_x, grad_y)

def div(v):
    """Returns the discrete divergence of a (n x m x 2) matrix v."""
    (v_x, v_y) = v # unpacking the 2 matrices within v
    (n,m,p) = v_x.shape
    
    div = np.zeros((n,m,p))

    # i in [2,n-1]
    for i in range (1,n-1):
        for j in range (1,m-1):
            div[i,j] = v_x[i,j]-v_x[i-1,j] + v_y[i,j]-v_y[i,j-1]
        div[i,0] = v_x[i,0]-v_x[i-1,0] + v_y[i,0]
        div[i,m-1] = v_x[i,m-1]-v_x[i-1,m-1] - v_y[i,m-2]
    # i = 1
    for j in range (1,m-1):
        div[0,j] = v_x[0,j] + v_y[0,j]-v_y[0,j-1]
    div[0,0] = v_x[0,0] + v_y[0,0]
    div[0,m-1] = v_x[0,m-1] - v_y[0,m-2]
    # i = n
    for j in range (1,m-1):
        div[n-1,j] = -v_x[n-2,j] + v_y[n-1,j]-v_y[n-1,j-1]
    div[n-1,0] = -v_x[n-2,0] + v_y[n-1,0]
    div[n-1,m-1] = -v_x[n-2,m-1] - v_y[n-1,m-2]

    return(div)

u = plt.imread('./robot_no_noise.jpg')
u = u / 255
res = div(grad(u))
res = np.mod(res,255)
res = res.astype(int)
plt.imshow(res)
plt.show()