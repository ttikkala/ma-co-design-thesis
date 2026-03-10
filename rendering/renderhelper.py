import cv2
import time

"""
This script contains some helper functions used in rendering.
"""

def grabFrame(env, camera_id=0):
    # Get RGB rendering of env
    rgbArr = env.physics.render(480, 600, camera_id)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)


def setupVideoWriter(env, folder_path='/home/tiia/thesis/videos/video_'):
    # Setup video writer - mp4 at 30 fps
    video_name = folder_path + time.ctime().replace(' ', '_') + '.mp4'
    frame = grabFrame(env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    return video


def grabFrameHD(env, camera_id=0):
    # Get RGB rendering of env
    rgbArr = env.physics.render(1080, 1350, camera_id)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)


def setupVideoWriterHD(env, folder_path='/home/tiia/thesis/videos/video_', video_stamp=None):
    # Setup video writer - mp4 at 30 fps
    if not video_stamp:
        video_name = folder_path + time.ctime().replace(' ', '_') + '.mp4'
    else:
        video_name = folder_path + video_stamp + '.mp4'
    
    frame = grabFrameHD(env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    return video

