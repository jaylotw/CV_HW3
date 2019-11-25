import argparse
import numpy as np
import cv2
import sys

debug = True
MIN_MATCH_COUNT = 15
DET_METHOD = 'ORB'

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
        detector = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
        
    if debug:
        print('detector: {}'.format(DET_METHOD))

    # Brute-force matcher
    if DET_METHOD == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher()
    
    # Find the keypoints and descriptors of marker
    kp_m, des_m = detector.detectAndCompute(template, None)    
    
    i = 0
    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {:03d}'.format(i))
        if ret:  # check whethere the frame is legal, i.e., there still exists a frame            
            
            # Find the keypoints and descriptors of frame
            kp_f, des_f = detector.detectAndCompute(frame, None)
            
            # Match the keypoints
            if DET_METHOD == 'ORB':
                matches = matcher.match(des_m, des_f)
                matches = sorted(matches, key = lambda x:x.distance)
            else:
                matches =  matcher.knnMatch(des_m, des_f, k=2)
                
            if debug:
                print('#keypoints in marker: %d, frame: %d' % (len(des_m), len(des_f)))
                print('#matches: %d' % (len(matches)))
                
            # Store all the good matches as per Lowe's ratio test
            if DET_METHOD == 'ORB':
                good = matches[:int(len(matches) * 0.2)]
            else:
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
            
                print('#good matches: %d' % (len(good)))
                cv2.imwrite("./debug/{}/match_{:03d}_{:03d}.jpg".format(DET_METHOD, i, len(good)), img_match)

            # Warp the reference image and paste it on the video frame
            frame = cv2.warpPerspective(src=ref_image, M=H, dsize=(film_w, film_h), dst=frame, borderMode=cv2.BORDER_TRANSPARENT)
            
            # Draw the mapped frame
            if debug:
                    cv2.imwrite("./debug/final/frame_{:03d}.png".format(i), frame)

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