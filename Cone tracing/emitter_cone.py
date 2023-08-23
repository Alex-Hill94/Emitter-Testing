import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

def source(n):
    np.random.seed()
    x_vals = np.random.rand(int(n)) - 0.5
    y_vals = np.zeros(len(x_vals))
    data = np.zeros((len(x_vals), 2))
    data[:,0] = x_vals
    data[:,1] = y_vals
    return data

def ang(n):
    np.random.seed()
    a = np.random.rand(n)
    return a*180.

def path(source, angle, length = 1000):
    x_offset = length*np.cos(angle)
    y_offset = abs(length*np.sin(angle))
    output = [source[0] - x_offset, source[1] + y_offset]
    return np.array(output)




def do_vectors_intersect(v1_start, v1_end, v2_start, v2_end):
    """
    Checks if two 2D vectors intersect.
    
    Parameters:
        v1_start (tuple): Starting point of the first vector (x1, y1).
        v1_end (tuple): Ending point of the first vector (x2, y2).
        v2_start (tuple): Starting point of the second vector (x3, y3).
        v2_end (tuple): Ending point of the second vector (x4, y4).
    
    Returns:
        bool: True if the vectors intersect, False otherwise.
    """
    def orientation(p, q, r):
        """
        Helper function to calculate the orientation of three points.
        
        Parameters:
            p (tuple): First point (x1, y1).
            q (tuple): Second point (x2, y2).
            r (tuple): Third point (x3, y3).
            
        Returns:
            int: The orientation value (0 if collinear, 1 if clockwise, -1 if counterclockwise).
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return -1  # Counterclockwise

    # Calculate orientations for all possible combinations of points
    o1 = orientation(v1_start, v1_end, v2_start)
    o2 = orientation(v1_start, v1_end, v2_end)
    o3 = orientation(v2_start, v2_end, v1_start)
    o4 = orientation(v2_start, v2_end, v1_end)

    # General case for intersecting vectors
    if o1 != o2 and o3 != o4:
        return True

    # Special cases for collinear vectors
    if o1 == 0 and is_point_on_vector(v1_start, v1_end, v2_start):
        return True
    if o2 == 0 and is_point_on_vector(v1_start, v1_end, v2_end):
        return True
    if o3 == 0 and is_point_on_vector(v2_start, v2_end, v1_start):
        return True
    if o4 == 0 and is_point_on_vector(v2_start, v2_end, v1_end):
        return True

    # Vectors do not intersect
    return False

def is_point_on_vector(start, end, point):
    """
    Helper function to check if a point lies on a vector.
    
    Parameters:
        start (tuple): Starting point of the vector (x1, y1).
        end (tuple): Ending point of the vector (x2, y2).
        point (tuple): The point to check (x, y).
        
    Returns:
        bool: True if the point lies on the vector, False otherwise.
    """
    if min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and \
       min(start[1], end[1]) <= point[1] <= max(start[1], end[1]):
        return True
    return False

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


line1 = [[-0.5, 0.0], [-0.5, 1.0]]
line2 = [[0.5, 0.0], [0.5, 1.0]]
line3 = [[-0.5, 0.0], [0.5, 0.0]]
line4 = [[-5.,5.0], [5.,5.0]]

N = 50000
v = source(N)
a = ang(N)

P = []
for i in range(0, N):
    p = path(v[i], a[i])
    P.append(p)

P = np.array(P)

X_VALS = []

fig, axs = plt.subplots(1,1, figsize = [8,4])

for i in tqdm(range(0, N)):
    intersect1 = do_vectors_intersect(v[i], P[i], line1[0], line1[1])
    intersect2 = do_vectors_intersect(v[i], P[i], line2[0], line2[1])
    intersection = (intersect1 == False) * (intersect2 == False)
    if intersection:
        axs.plot([v[i][0], P[i][0]], [v[i][1], P[i][1]], color = 'k', alpha = 0.2)
        IP = line_intersection((v[i], P[i]), (line4[0], line4[1]))
        plt.scatter(IP[0], IP[1], marker = 'x', color = 'blue', s = 0.1, alpha = 0.1)
        X_VALS.append(IP[0])
#        else:
#            print(i, 'escapes, but no intersection')
    else:
        axs.plot([v[i][0], P[i][0]], [v[i][1], P[i][1]], color = 'red', alpha = 0.1, ls = '-', lw = 0.1)
axs.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], color = 'blue', lw = 3)
axs.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], color = 'blue', lw = 3)
axs.plot([line3[0][0], line3[1][0]], [line3[0][1], line3[1][1]], color = 'blue', lw = 3)
axs.plot([line4[0][0], line4[1][0]], [line4[0][1], line4[1][1]], color = 'blue', lw = 3)
axs.set_xlim([-20, 20])
axs.set_ylim([0, 20])
plt.savefig('diagram.png')

X_VALS = np.array(X_VALS)
plt.figure()
plt.hist(X_VALS, bins = 100)
plt.savefig('plot.png')

'''
v1_start = (0, 0)
v1_end = (4, 4)
v2_start = (2, 0)
v2_end = (2, 4)

I = []
for i in range(0, N):
    intersect1 = do_vectors_intersect(v[i], P[i], line1[0], line1[1])
    intersect2 = do_vectors_intersect(v[i], P[i], line2[0], line2[1])
    intersection = (intersect1 == False) * (intersect2 == False)
    if intersection:
        IP = find_intersection_point(v[i], P[i], line4[0], line4[1])
        I.append(IP)

print(intersection_point)  # Output: (2.0, 2.0)
'''