import argparse
import numpy as np
import cv2
import sys

def main(ref_image, template ,video):
    ref_image = cv2.imread(ref_imag, cv2.IMREAD_GRAYSCALE)  ## load gray if you need.
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)  ## load gray if you need.
    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))
    i = 0
    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}'.format(i))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            ## TODO: homography transform, feature detection, ransanc, etc.


            videowriter.write(frame)

        else:
            break
            
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./ar_marker.mp4')
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = parser.parse_args()  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)