import numpy as np
import cv2
import time


'''
Coordinate System

    0/0---X--->
     |
     |
     Y
     |
     |
     v
'''

def coordinate_u(img):
    '''
    Each column of the coordinate matrix consists of [x, y, 1]^T
    
    Use the same coordinate system used by OpenCV
    (x: horizontal, y: vertical).
    '''
    h, w, ch = img.shape   
    img_size = h*w
    
    coordinate = np.ones((3, img_size), dtype=int)
    v1 = np.arange(h)
    v2 = np.arange(w)
    coordinate[0] = np.tile(v2, h)
    coordinate[1] = np.repeat(v1, w, axis=0)

    return coordinate


def homography_transform(H, u):
    '''
    Input H and u. Return a normalized v.
    '''
    v = np.dot(H, u)    # v=H*u
    v *= 1/v[2]         # Normalize v
    return v
    
    
def coordinate_2d(vec):
    '''
    Input a 3 by n matrix and return a n by 2 matrix with each row corresponds to (x, y) position
    '''
    dest = np.delete(vec, obj=2, axis=0).T
    return dest


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v, method=2):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
        return None
    
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))

    # Method 1
    if(method==1):
        # Build vector b
        b = v.reshape(-1, 1)

        # Build matrix A
        A = np.zeros((2*N, 8))
        for i in range(N):
            A[2*i, 0:2] = u[i]
            A[2*i, 2] = 1
            A[2*i, 6] = - u[i, 0] * v[i, 0]
            A[2*i, 7] = - u[i, 1] * v[i, 0]

            A[2*i + 1, 3:5] = u[i]
            A[2*i + 1, 5] = 1
            A[2*i + 1, 6] = - u[i, 0] * v[i, 1]
            A[2*i + 1, 7] = - u[i, 1] * v[i, 1]

        # Solve the linear matrix equation Ax=b
        x = np.linalg.solve(A, b)

        # Reshape the solution and fill in the value of h33=1
        H = np.resize(x, (3, 3))
        H[2, 2] = 1

    # Method 2
    elif(method==2):
        # Build matrix A
        A = np.zeros((2*N, 9))
        for i in range(N):
            A[2*i, 0:2] = u[i]
            A[2*i, 2] = 1
            A[2*i, 6] = - u[i, 0] * v[i, 0]
            A[2*i, 7] = - u[i, 1] * v[i, 0]
            A[2*i, 8] = - v[i, 0]

            A[2*i + 1, 3:5] = u[i]
            A[2*i + 1, 5] = 1
            A[2*i + 1, 6] = - u[i, 0] * v[i, 1]
            A[2*i + 1, 7] = - u[i, 1] * v[i, 1]
            A[2*i + 1, 8] = - v[i, 1]
        
        # Compute SVD of A
        _, _, vh = np.linalg.svd(A)

        # H is the last column of V
        V = vh.T
        H = np.resize(V[:, 8], (3, 3))

    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    img_size = h*w
    img_corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    H = solve_homography(img_corners, corners)

    u = coordinate_u(img)
    v = homography_transform(H, u)
    v = coordinate_2d(v)
    u = coordinate_2d(u)
    v = v.astype(int)

    for i in range(img_size):
        pixel = img[u[i, 1], u[i, 0], :]
        canvas[v[i, 1], v[i, 0], :] = pixel


def back_warping(src_img, project, corner):
    h, w, ch = project.shape
    project_size = h*w
    pro_corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    H = solve_homography(pro_corners, corner)

    u = coordinate_u(project)
    v = homography_transform(H, u)
    v = coordinate_2d(v)
    u = coordinate_2d(u)

    for i in range(project_size):
        project[u[i, 1], u[i, 0]] = bilinear(src_img, v[i, 1], v[i, 0])


def bilinear(src_img, t_y, t_x):
    if t_x < 0 or t_y < 0 or t_x >= src_img.shape[1]-1 or t_y >= src_img.shape[0]-1:
        return np.zeros(3)
    
    x_left = round(t_x - int(t_x), 4)
    x_right = 1 - x_left
    y_low = round(t_y - int(t_y), 4)
    y_high = 1 - y_low

    int_x, int_y = int(round(t_x)), int(round(t_y))

    try:
        img_low_left = x_left * y_low * src_img[int_y, int_x]
        img_low_right = x_right * y_low * src_img[int_y, int_x + 1]
        img_high_left = x_left * y_high * src_img[int_y + 1, int_x]
        img_high_right = x_right * y_high * src_img[int_y + 1, int_x + 1]
        img_sum = img_low_left + img_low_right + img_high_left + img_high_right
    except IndexError:
        return np.zeros(3)

    return img_sum


def main():
    # Part 1
    print("========== PART 1 ===========")
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpeg')
    img2 = cv2.imread('./input/kuo.jpg')
    img3 = cv2.imread('./input/haung.jpg')
    img4 = cv2.imread('./input/tsai.jpg')
    img5 = cv2.imread('./input/han.jpg')

    canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])
    
    transform(img1, canvas, canvas_corners1)
    transform(img2, canvas, canvas_corners2)
    transform(img3, canvas, canvas_corners3)
    transform(img4, canvas, canvas_corners4)
    transform(img5, canvas, canvas_corners5)
    
    cv2.imwrite('part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 2
    print("========== PART 2 ===========")
    ts = time.time()
    img = cv2.imread('./input/QR_code.jpg')

    output2 = np.zeros((200, 200, 3))
    # corner_screen = np.array([[778, 364], [2280, 541], [1578, 2895], [2558, 1662]])
    corner_QR = np.array([[1979, 1238], [2041, 1211], [2025, 1397], [2084, 1365]])
    back_warping(img, output2, corner_QR)

    cv2.imwrite('part2.png', output2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 3
    print("========== PART 3 ===========")
    ts = time.time()
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    
    output3 = np.zeros((300, 500, 3))
    corners_crosswalk = np.array([[160, 129], [563, 129], [0, 286], [723, 286]])
    back_warping(img_front, output3, corners_crosswalk)

    cv2.imwrite('part3.png', output3)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

if __name__ == '__main__':
    main()
