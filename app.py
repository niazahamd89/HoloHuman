from camera import Camera
from video import generate_video_results
from humvis import HumVis

import torch
import cv2
import os
import numpy as np
from PIL import Image
import base64
import io

from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------
model = HumVis(backbone_name='hrnet32')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)

camera = Camera(model)

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_DIR'] = 'static/uploads'

# Sub-folder inside UPLOAD_DIR where prediction images are saved
RESULTS_SUBDIR = 'results'

# Sub-folder inside UPLOAD_DIR where video prediction results are saved
VIDEO_RESULTS_SUBDIR = 'video-results'

print('index')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _save_result_image(frame_bgr: np.ndarray,
                       base_name: str,
                       suffix: str,
                       results_dir: str) -> str:
    """
    Save a BGR numpy frame to disk as JPEG.

    Returns the path relative to the static folder (usable in <img src>).
    """
    _ensure_dir(results_dir)
    filename  = f"{base_name}_{suffix}.jpg"
    full_path = os.path.join(results_dir, filename)
    cv2.imwrite(full_path, frame_bgr)
    rel_path  = os.path.relpath(full_path, start='static')
    return rel_path


def _frame_to_base64(frame_bgr: np.ndarray) -> str:
    """Convert a BGR numpy frame to a base-64 JPEG string for inline HTML."""
    frame_rgb = frame_bgr[:, :, ::-1]
    im  = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _make_joint_frame(original: np.ndarray,
                      frame_pose: np.ndarray,
                      frame_mask: np.ndarray,
                      mask_alpha: float = 0.55,
                      diff_thresh: int = 25) -> np.ndarray:
    """
    Build a single joint visualisation: segmentation overlay + pose skeleton.

    The model returns two independent renderings (pose drawn on the original,
    mask drawn on the original). To combine them without touching the model:

    1. Blend the segmentation result with the original image.
    2. Detect which pixels the pose renderer changed (skeleton / keypoints)
       by diffing frame_pose against the original.
    3. Paint those skeleton pixels on top of the blended segmentation.

    Parameters
    ----------
    original    : raw uploaded frame (BGR)
    frame_pose  : pose visualisation returned by the model (BGR)
    frame_mask  : segmentation visualisation returned by the model (BGR)
    mask_alpha  : weight of the segmentation overlay (0..1)
    diff_thresh : per-channel difference threshold used to isolate the
                  skeleton drawing from the pose frame

    Returns
    -------
    np.ndarray : combined BGR frame
    """
    # Guard against size mismatches between the model outputs
    h, w = original.shape[:2]
    if frame_pose.shape[:2] != (h, w):
        frame_pose = cv2.resize(frame_pose, (w, h))
    if frame_mask.shape[:2] != (h, w):
        frame_mask = cv2.resize(frame_mask, (w, h))

    # 1) Segmentation blended over the original
    joint = cv2.addWeighted(frame_mask, mask_alpha,
                            original,   1.0 - mask_alpha, 0)

    # 2) Pixels where the pose renderer drew something (skeleton, joints)
    diff          = cv2.absdiff(frame_pose, original)
    skeleton_mask = np.any(diff > diff_thresh, axis=2)

    # Slightly dilate so thin skeleton lines stay visible after blending
    skeleton_mask = cv2.dilate(
        skeleton_mask.astype(np.uint8),
        np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    # 3) Skeleton on top of the segmentation overlay
    joint[skeleton_mask] = frame_pose[skeleton_mask]
    return joint


def _enhance_part_contrast(frame_part: np.ndarray,
                           original: np.ndarray = None,
                           sat_gain: float = 1.7,
                           contrast: float = 1.35,
                           diff_thresh: int = 20) -> np.ndarray:
    """
    Increase the colour contrast of the part-segmentation visualisation.

    The part colours are made more vivid by boosting saturation in HSV space
    and stretching contrast around the midpoint. If the raw input frame is
    provided, the enhancement is applied ONLY to the pixels the part renderer
    actually coloured (found by diffing against the original), so the
    background keeps its natural appearance.

    Parameters
    ----------
    frame_part  : part segmentation visualisation (BGR)
    original    : raw input frame (BGR), optional
    sat_gain    : saturation multiplier (>1 = more vivid colours)
    contrast    : contrast gain around the midpoint (>1 = more contrast)
    diff_thresh : per-channel threshold to locate the coloured overlay
    """
    # Boost saturation
    hsv = cv2.cvtColor(frame_part, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Stretch contrast around the midpoint (128)
    enhanced = cv2.convertScaleAbs(enhanced,
                                   alpha=contrast,
                                   beta=128.0 * (1.0 - contrast))

    if original is None:
        return enhanced

    # Restrict the enhancement to the coloured part regions only
    h, w = frame_part.shape[:2]
    if original.shape[:2] != (h, w):
        original = cv2.resize(original, (w, h))
    part_region = np.any(cv2.absdiff(frame_part, original) > diff_thresh,
                         axis=2)
    out = frame_part.copy()
    out[part_region] = enhanced[part_region]
    return out


def _blend_joint_fallback(frame_pose: np.ndarray,
                          frame_mask: np.ndarray) -> np.ndarray:
    """
    Combine pose and segmentation frames when the raw input frame is not
    available (live stream). A plain 50/50 blend keeps both the mask overlay
    and the skeleton visible.
    """
    h, w = frame_pose.shape[:2]
    if frame_mask.shape[:2] != (h, w):
        frame_mask = cv2.resize(frame_mask, (w, h))
    return cv2.addWeighted(frame_mask, 0.5, frame_pose, 0.5, 0)


def _get_camera_raw_frame():
    """
    Try to fetch the raw (unannotated) webcam frame from the Camera object.
    Different Camera implementations expose it under different names; if none
    exists, return None and the caller falls back to a plain blend.
    """
    for attr in ('get_raw_frame', 'get_input_frame', 'get_frame'):
        fn = getattr(camera, attr, None)
        if callable(fn):
            try:
                frame = fn()
                if isinstance(frame, np.ndarray):
                    return frame
            except Exception:
                pass
    return None


def generate_joint_video(filepath: str, output_dir: str) -> tuple:
    """
    Run the model on every frame of an uploaded video and write TWO videos:

      <name>_joint.mp4       — pose skeleton + person segmentation
      <name>_joint_part.mp4  — pose skeleton + body-part segmentation
                               (with contrast-enhanced part colours)

    Both are produced in a single pass so inference runs only once per frame.

    Returns (joint_filename, joint_part_filename), relative to output_dir.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {filepath}')

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name       = os.path.splitext(os.path.basename(filepath))[0]
    joint_name      = f"{base_name}_joint.mp4"
    joint_part_name = f"{base_name}_joint_part.mp4"
    joint_path      = os.path.join(output_dir, joint_name)
    joint_part_path = os.path.join(output_dir, joint_part_name)

    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    writer      = cv2.VideoWriter(joint_path,      fourcc, fps, (width, height))
    writer_part = cv2.VideoWriter(joint_part_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_pose, frame_mask, frame_part, _ = model.get_analysis_results(frame)

        # Pose + person segmentation
        joint = _make_joint_frame(frame, frame_pose, frame_mask)
        if joint.shape[:2] != (height, width):
            joint = cv2.resize(joint, (width, height))
        writer.write(joint)

        # Pose + part segmentation (contrast-enhanced, matching the image path)
        frame_part = _enhance_part_contrast(frame_part, original=frame)
        joint_part = _make_joint_frame(frame, frame_pose, frame_part)
        if joint_part.shape[:2] != (height, width):
            joint_part = cv2.resize(joint_part, (width, height))
        writer_part.write(joint_part)

        frame_idx += 1
        if frame_idx % 25 == 0:
            print(f'[INFO] Joint videos: processed {frame_idx} frames')

    cap.release()
    writer.release()
    writer_part.release()
    print(f'[INFO] Joint video saved → {joint_path} ({frame_idx} frames)')
    print(f'[INFO] Joint part video saved → {joint_part_path} ({frame_idx} frames)')
    return joint_name, joint_part_name


# ---------------------------------------------------------------------------
# Routes — index
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('demo_index.html')


# ---------------------------------------------------------------------------
# Routes — live stream
# ---------------------------------------------------------------------------
@app.route('/live/')
def live_index():
    return render_template('live_index.html')


@app.route('/live/receiveimage', methods=["POST"])
def receiveimage():
    if request.method == "POST":
        json_data = request.get_json()
        str_image = json_data.get("imgData")
        camera.enqueue_input(str_image)
    return Response('upload')


def gen_stream(output_type='pose'):
    while True:
        if output_type == 'pose':
            frame = camera.get_pose_frame()
        elif output_type == 'mask':
            frame = camera.get_mask_frame()
        elif output_type == 'part':
            frame = camera.get_part_frame()
            frame = _enhance_part_contrast(frame,
                                           original=_get_camera_raw_frame())
        elif output_type == 'mesh':
            frame = camera.get_mesh_frame()
        elif output_type == 'joint':
            frame_pose = camera.get_pose_frame()
            frame_mask = camera.get_mask_frame()
            raw = _get_camera_raw_frame()
            if raw is not None:
                frame = _make_joint_frame(raw, frame_pose, frame_mask)
            else:
                frame = _blend_joint_fallback(frame_pose, frame_mask)
        elif output_type == 'joint_part':
            frame_pose = camera.get_pose_frame()
            frame_part = camera.get_part_frame()
            raw = _get_camera_raw_frame()
            frame_part = _enhance_part_contrast(frame_part, original=raw)
            if raw is not None:
                frame = _make_joint_frame(raw, frame_pose, frame_part)
            else:
                frame = _blend_joint_fallback(frame_part, frame_pose)
        else:
            raise ValueError(f'Unknown output type: {output_type}')
        assert isinstance(frame, np.ndarray)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/live/video_pose_feed')
def video_pose_feed():
    return Response(gen_stream('pose'), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/live/video_mask_feed')
def video_mask_feed():
    return Response(gen_stream('mask'), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/live/video_part_feed')
def video_part_feed():
    return Response(gen_stream('part'), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/live/video_mesh_feed')
def video_mesh_feed():
    return Response(gen_stream('mesh'), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/live/video_joint_feed')
def video_joint_feed():
    """Live webcam feed with combined pose + segmentation visualisation."""
    return Response(gen_stream('joint'), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/live/video_joint_part_feed')
def video_joint_part_feed():
    """Live webcam feed with combined pose + part segmentation visualisation."""
    return Response(gen_stream('joint_part'), mimetype='multipart/x-mixed-replace;boundary=frame')


# ---------------------------------------------------------------------------
# Routes — image upload
# ---------------------------------------------------------------------------
@app.route('/upload/', methods=['GET', 'POST'])
def upload_index():
    if request.method == 'POST':
        is_video = False
        if 'image' in request.files:
            file = request.files['image']
        elif 'video' in request.files:
            file = request.files['video']
            is_video = True
        else:
            raise ValueError('Unknown upload type')

        filename = secure_filename(file.filename)
        _ensure_dir(app.config['UPLOAD_DIR'])
        filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
        if not os.path.exists(filepath):
            file.save(filepath)

        if is_video:
            return redirect(url_for('.upload_video_result', filename=filename))
        else:
            return redirect(url_for('.upload_result', filename=filename))

    return render_template('upload_index.html')


@app.route('/upload/result', methods=['GET'])
def upload_result():
    filename  = request.args.get('filename')
    filepath  = os.path.join(app.config['UPLOAD_DIR'], filename)
    base_name = os.path.splitext(filename)[0]

    # ── Run inference ─────────────────────────────────────────────────────────
    frame = cv2.imread(filepath)
    frame_pose, frame_mask, frame_part, frame_mesh = model.get_analysis_results(frame)

    # ── Boost colour contrast of the part segmentation ────────────────────────
    frame_part = _enhance_part_contrast(frame_part, original=frame)

    # ── Joint pose + segmentation visualisation (shown in the image slot) ─────
    frame_joint = _make_joint_frame(frame, frame_pose, frame_mask)

    # ── Joint pose + PART segmentation visualisation ──────────────────────────
    # Same composition, but with the body-part segmentation as the overlay.
    frame_joint_part = _make_joint_frame(frame, frame_pose, frame_part)

    # ── Save all results to disk ──────────────────────────────────────────────
    results_dir = os.path.join(app.config['UPLOAD_DIR'], RESULTS_SUBDIR)
    saved_paths = {}
    all_results = [
        ('joint',      frame_joint),
        ('joint_part', frame_joint_part),
        ('pose',       frame_pose),
        ('mask',       frame_mask),
        ('part',       frame_part),
        ('mesh',       frame_mesh),
    ]
    for label, result_frame in all_results:
        rel = _save_result_image(result_frame, base_name, label, results_dir)
        saved_paths[label] = rel
        print(f'[INFO] Saved {label} result → static/{rel}')

    # ── Base64 for inline HTML display ────────────────────────────────────────
    b64 = {label: _frame_to_base64(frm) for label, frm in all_results}

    return render_template(
        'upload_result.html',
        filename    = filename,
        # The "input image" slot now shows the joint pose + segmentation
        # visualisation instead of the raw uploaded image.
        input_img   = b64['joint'],
        input_saved = saved_paths['joint'],
        # Also expose the joint result under its own names (kept for
        # compatibility if the template references joint_img elsewhere)
        joint_img        = b64['joint'],
        joint_saved      = saved_paths['joint'],
        # Joint pose + part segmentation result
        joint_part_img   = b64['joint_part'],
        joint_part_saved = saved_paths['joint_part'],
        # Individual results (inline base64)
        pose_img    = b64['pose'],
        mask_img    = b64['mask'],
        part_img    = b64['part'],
        mesh_img    = b64['mesh'],
        # Saved file paths (for download links)
        pose_saved  = saved_paths['pose'],
        mask_saved  = saved_paths['mask'],
        part_saved  = saved_paths['part'],
        mesh_saved  = saved_paths['mesh'],
    )


# ---------------------------------------------------------------------------
# Routes — video upload
# ---------------------------------------------------------------------------
@app.route('/upload/videoresult', methods=['GET'])
def upload_video_result():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_DIR'], filename)

    # All video results go into <UPLOAD_DIR>/video-results/
    video_results_dir = os.path.join(app.config['UPLOAD_DIR'],
                                     VIDEO_RESULTS_SUBDIR)
    _ensure_dir(video_results_dir)

    ret_filenames = generate_video_results(
        filepath, model, video_results_dir)

    print(f'[INFO] Video results saved to {video_results_dir}: {ret_filenames}')

    # Joint pose + segmentation AND pose + part segmentation videos
    joint_video, joint_part_video = generate_joint_video(
        filepath, video_results_dir)

    # Paths relative to the static folder, used by the template
    def _rel(name: str) -> str:
        return os.path.relpath(os.path.join(video_results_dir, name),
                               start='static')

    return render_template(
        'upload_video_result.html',
        filename         = filename,
        # The "input video" slot shows the joint pose + segmentation result
        input_video      = _rel(joint_video),
        joint_video      = _rel(joint_video),
        # Pose + part segmentation video
        joint_part_video = _rel(joint_part_video),
        pose_video       = _rel(ret_filenames[0]),
        mask_video       = _rel(ret_filenames[1]),
        part_video       = _rel(ret_filenames[2]),
        mesh_video       = _rel(ret_filenames[3]),
    )


# ---------------------------------------------------------------------------
print('app run')
app.run(host='0.0.0.0', port=65432, debug=True, threaded=True)



######old code#################

# from camera import Camera
# from video import generate_video_results
# from humvis import HumVis
#
# import torch
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import base64
# import io
#
# from flask import Flask, render_template, Response, request, redirect, url_for
# from werkzeug.utils import secure_filename
#
# # ---------------------------------------------------------------------------
# # Model initialisation
# # ---------------------------------------------------------------------------
# model = HumVis(backbone_name='hrnet32')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.model.to(device)
#
# camera = Camera(model)
#
# # ---------------------------------------------------------------------------
# # App config
# # ---------------------------------------------------------------------------
# app = Flask(__name__)
# app.config['UPLOAD_DIR'] = 'static/uploads'
#
# # Sub-folder inside UPLOAD_DIR where prediction images are saved
# RESULTS_SUBDIR = 'results'
#
# print('index')
#
#
# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
# def _ensure_dir(path: str):
#     """Create directory (and parents) if it does not exist."""
#     os.makedirs(path, exist_ok=True)
#
#
# def _save_result_image(frame_bgr: np.ndarray,
#                         base_name: str,
#                         suffix: str,
#                         results_dir: str) -> str:
#     """
#     Save a BGR numpy frame to disk as JPEG.
#
#     Parameters
#     ----------
#     frame_bgr   : BGR image array (H, W, 3)
#     base_name   : input filename without extension, e.g. 'my_photo'
#     suffix      : label for this result, e.g. 'pose', 'mask', 'part', 'mesh'
#     results_dir : absolute or relative directory to save into
#
#     Returns
#     -------
#     str : relative path to the saved file (suitable for <img src="...">)
#     """
#     _ensure_dir(results_dir)
#     filename  = f"{base_name}_{suffix}.jpg"
#     full_path = os.path.join(results_dir, filename)
#     cv2.imwrite(full_path, frame_bgr)
#     # Return path relative to the static folder so Flask can serve it
#     rel_path  = os.path.relpath(full_path, start='static')
#     return rel_path
#
#
# def _frame_to_base64(frame_bgr: np.ndarray) -> str:
#     """Convert a BGR numpy frame to a base-64 JPEG string for inline HTML."""
#     frame_rgb = frame_bgr[:, :, ::-1]
#     im  = Image.fromarray(frame_rgb)
#     buf = io.BytesIO()
#     im.save(buf, "JPEG")
#     return base64.b64encode(buf.getvalue()).decode('utf-8')
#
#
# # ---------------------------------------------------------------------------
# # Routes — index
# # ---------------------------------------------------------------------------
# @app.route('/')
# def index():
#     return render_template('demo_index.html')
#
#
# # ---------------------------------------------------------------------------
# # Routes — live stream
# # ---------------------------------------------------------------------------
# @app.route('/live/')
# def live_index():
#     return render_template('live_index.html')
#
#
# @app.route('/live/receiveimage', methods=["POST"])
# def receiveimage():
#     if request.method == "POST":
#         json_data   = request.get_json()
#         str_image   = json_data.get("imgData")
#         camera.enqueue_input(str_image)
#     return Response('upload')
#
#
# def gen_stream(output_type='pose'):
#     while True:
#         if output_type == 'pose':
#             frame = camera.get_pose_frame()
#         elif output_type == 'mask':
#             frame = camera.get_mask_frame()
#         elif output_type == 'part':
#             frame = camera.get_part_frame()
#         elif output_type == 'mesh':
#             frame = camera.get_mesh_frame()
#         else:
#             raise ValueError(f'Unknown output type: {output_type}')
#         assert isinstance(frame, np.ndarray)
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\b')
#
#
# @app.route('/live/video_pose_feed')
# def video_pose_feed():
#     return Response(gen_stream('pose'), mimetype='multipart/x-mixed-replace;boundary=frame')
#
#
# @app.route('/live/video_mask_feed')
# def video_mask_feed():
#     return Response(gen_stream('mask'), mimetype='multipart/x-mixed-replace;boundary=frame')
#
#
# @app.route('/live/video_part_feed')
# def video_part_feed():
#     return Response(gen_stream('part'), mimetype='multipart/x-mixed-replace;boundary=frame')
#
#
# @app.route('/live/video_mesh_feed')
# def video_mesh_feed():
#     return Response(gen_stream('mesh'), mimetype='multipart/x-mixed-replace;boundary=frame')
#
#
# # ---------------------------------------------------------------------------
# # Routes — image upload
# # ---------------------------------------------------------------------------
# @app.route('/upload/', methods=['GET', 'POST'])
# def upload_index():
#     if request.method == 'POST':
#         is_video = False
#         if 'image' in request.files:
#             file = request.files['image']
#         elif 'video' in request.files:
#             file = request.files['video']
#             is_video = True
#         else:
#             raise ValueError('Unknown upload type')
#
#         filename = secure_filename(file.filename)
#         _ensure_dir(app.config['UPLOAD_DIR'])
#         filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
#         if not os.path.exists(filepath):
#             file.save(filepath)
#
#         if is_video:
#             return redirect(url_for('.upload_video_result', filename=filename))
#         else:
#             return redirect(url_for('.upload_result', filename=filename))
#
#     return render_template('upload_index.html')
#
#
# @app.route('/upload/result', methods=['GET'])
# def upload_result():
#     filename  = request.args.get('filename')
#     filepath  = os.path.join(app.config['UPLOAD_DIR'], filename)
#     base_name = os.path.splitext(filename)[0]
#
#     # ── Run inference ─────────────────────────────────────────────────────────
#     frame = cv2.imread(filepath)
#     frame_pose, frame_mask, frame_part, frame_mesh = model.get_analysis_results(frame)
#
#     # ── Save all four results to disk ─────────────────────────────────────────
#     results_dir = os.path.join(app.config['UPLOAD_DIR'], RESULTS_SUBDIR)
#     saved_paths = {}
#     for label, result_frame in [
#         ('pose', frame_pose),
#         ('mask', frame_mask),
#         ('part', frame_part),
#         ('mesh', frame_mesh),
#     ]:
#         rel = _save_result_image(result_frame, base_name, label, results_dir)
#         saved_paths[label] = rel
#         print(f'[INFO] Saved {label} result → static/{rel}')
#
#     # ── Also encode as base64 for inline HTML display ─────────────────────────
#     b64 = {label: _frame_to_base64(frm)
#            for label, frm in [
#                ('pose', frame_pose),
#                ('mask', frame_mask),
#                ('part', frame_part),
#                ('mesh', frame_mesh),
#            ]}
#
#     return render_template(
#         'upload_result.html',
#         filename        = filename,
#         # Inline base64 images (shown directly in the page)
#         pose_img        = b64['pose'],
#         mask_img        = b64['mask'],
#         part_img        = b64['part'],
#         mesh_img        = b64['mesh'],
#         # Saved file paths (can be used for download links in the template)
#         pose_saved      = saved_paths['pose'],
#         mask_saved      = saved_paths['mask'],
#         part_saved      = saved_paths['part'],
#         mesh_saved      = saved_paths['mesh'],
#     )
#
#
# # ---------------------------------------------------------------------------
# # Routes — video upload
# # ---------------------------------------------------------------------------
# @app.route('/upload/videoresult', methods=['GET'])
# def upload_video_result():
#     filename = request.args.get('filename')
#     filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
#
#     # generate_video_results saves result videos and returns their filenames
#     ret_filenames = generate_video_results(
#         filepath, model, app.config['UPLOAD_DIR'])
#
#     print(f'[INFO] Video results saved: {ret_filenames}')
#
#     return render_template(
#         'upload_video_result.html',
#         filename   = filename,
#         pose_video = ret_filenames[0],
#         mask_video = ret_filenames[1],
#         part_video = ret_filenames[2],
#         mesh_video = ret_filenames[3],
#     )
#
#
# # ---------------------------------------------------------------------------
# print('app run')
# app.run(host='0.0.0.0', port=65432, debug=True, threaded=True)