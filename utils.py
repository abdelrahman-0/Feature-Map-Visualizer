import cv2

def export_video(frames, title, fps=10):
    height, width = frames[0].shape[:2]
    
    # Export video as mp4
    if '.mp4' in title:
        full_title = title
    else:
        full_title = title + '.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    # Write frames to output video
    video = cv2.VideoWriter(full_title, fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()
    cv2.destroyAllWindows()