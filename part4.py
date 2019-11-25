import argparse
import numpy as np
import cv2
import sys
from main import coordinate_u, homography_transform, coordinate_2d

debug = True
MIN_MATCH_COUNT = 15
DET_METHOD = 'SURF'

def main(ref_image, template ,video):
    ref_image = cv2.imread(ref_image)  ## load gray if you need.
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)  ## load gray if you need.
    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))

    # Resize the reference image to match the size of marker
    ref_image = cv2.resize(ref_image, template.shape)

    # Initiate detector
    if DET_METHOD == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    elif DET_METHOD == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()
    elif DET_METHOD == 'ORB':
        detector = cv2.ORB_create()
        
    if debug:
        print('detector: {}'.format(DET_METHOD))

    # Brute-force matcher
    matcher = cv2.BFMatcher()
    # Find the keypoints and descriptors of marker
    kp_m, des_m = detector.detectAndCompute(template, None)    
    
    i = 0
    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {:04d}'.format(i))
        if ret:  # check whethere the frame is legal, i.e., there still exists a frame            
            
            # Find the keypoints and descriptors of frame
            kp_f, des_f = detector.detectAndCompute(frame, None)
            
            if debug:
                print('#keypoints in marker: %d, frame: %d' % (len(des_m), len(des_f)))
            
            # Match the keypoints
            matches =  matcher.knnMatch(des_m, des_f, k=2)
                
            # Store all the good matches as per Lowe's ratio test
            good = list()
            for (m, n) in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            
            # Run RANSAC and get the homography
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([ kp_m[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_f[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
            else:
                print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
                matchesMask = None
            
            # Draw matches
            if debug:
                draw_params = dict(matchesMask=matchesMask, flags=2)
            
                img_match = cv2.drawMatches(template, kp_m, frame, kp_f, good, None, **draw_params)
            
                print('Final matches: %d' % (len(good)))
                cv2.imwrite("./debug/{}/match_{:04d}_{:03d}.jpg".format(DET_METHOD, i, len(good)), img_match)

            # Warp the reference image and paste it on the video frame
            test = cv2.warpPerspective(ref_image, H, (film_w, film_h))
            cv2.imwrite("test.jpg", test)

            videowriter.write(frame)
            i += 1

        else:
            break
            
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Marker based AR')
    parser.add_argument("--video_path", dest="video_path", default='./input/ar_marker.mp4')
    args = parser.parse_args()
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = args.video_path  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)