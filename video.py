import cv2
import os
import time

def get_video_info(in_file):
    # Open the video capture stream
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), f'Cannot capture source: {in_file}'

    # Get video properties
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width of the frames
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height of the frames
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))  # Get the codec
    fps = stream.get(cv2.CAP_PROP_FPS)  # Get the frames per second

    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': (width, height)}
    stream.release()

    return stream, videoinfo, datalen

def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'MP4V'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print(f"Unknown video format {ext}, using .mp4 instead.")
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

def draw_fps(frame, fps):
    """Draw FPS counter on the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def generate_video_results(input_video, model, output_path):
    video_name = input_video
    video_basename = os.path.basename(video_name).split('.')[0]
    output_types = ['pose', 'mask', 'part', 'mesh']
    final_output_video_list = []
    ret_filename = []

    # Get video info (including frame width and height)
    _, info, _ = get_video_info(video_name)
    fixed_width, fixed_height = info['frameSize']  # Automatically retrieve width and height

    # Create video writer streams for each output type
    for out in output_types:
        ret_filename.append(f'{video_basename}_{out}.mp4')
        savepath = os.path.join(output_path, f'{video_basename}_{out}.mp4')
        info['savepath'] = savepath
        info['fourcc'] = cv2.VideoWriter_fourcc(*'mp4v')

        write_stream = cv2.VideoWriter(
            info['savepath'], info['fourcc'], info['fps'], info['frameSize']
        )

        assert write_stream.isOpened(), f'Cannot open video stream for {out}'
        final_output_video_list.append(write_stream)

    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), f'Cannot open input video: {video_name}'

    cnt = 0
    start_time = time.time()

    while True:
        frameState, frame = cap.read()
        if not frameState:
            break
        cnt += 1

        # Resize the frame based on the automatically retrieved width and height
        frame = cv2.resize(frame, (fixed_width, fixed_height))
        frame_pose, frame_mask, frame_part, frame_mesh = model.get_analysis_results(frame, tracking=True)
        if any(f is None for f in [frame_pose, frame_mask, frame_part, frame_mesh]):
            print(f"Skipping frame {cnt}: One of the analysis results is None.")
            continue

        # Calculate and draw FPS
        #elapsed_time = time.time() - start_time
        #fps = cnt / elapsed_time if elapsed_time > 0 else 0
        #draw_fps(frame_pose, fps * 5)
        #draw_fps(frame_mask, fps* 5)
        #draw_fps(frame_part, fps* 5)
        #draw_fps(frame_mesh, fps* 5)

        final_output_video_list[0].write(frame_pose)
        final_output_video_list[1].write(frame_mask)
        final_output_video_list[2].write(frame_part)
        final_output_video_list[3].write(frame_mesh)

    #print(f"Total frames processed: {cnt}")
    #print('FPS:', fps)

    cap.release()
    for stream in final_output_video_list:
        stream.release()

    convert_filenames = []
    for gen_file in ret_filename:
        savepath = os.path.join(output_path, gen_file)
        basename = os.path.basename(savepath).split('.')[0]
        newpath = os.path.join(output_path, f'{basename}_final.mp4')
        if os.path.exists(savepath) and os.path.getsize(savepath) > 0:
            os.system(f'ffmpeg -i {savepath} -vcodec libx264 {newpath}')
            convert_filenames.append(f'{basename}_final.mp4')
        else:
            print(f"Skipping conversion for {savepath}: file not found or empty.")

    return convert_filenames
