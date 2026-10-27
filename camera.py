import os
import atexit
import threading
from time import sleep, time
import numpy as np
import cv2
import base64


class Camera(object):
    def __init__(
        self,
        model,
        save_webcam_results=True,
        video_fps=25,               # fallback if auto-detect not ready
        prefer_mp4=True,
        out_name_prefix="webcam",
        save_original=True,         # also save the raw camera frames

        # --- auto input-fps settings ---
        match_input_fps=True,        # if True, detect input FPS from enqueue timing
        fps_window=30,               # how many enqueue deltas to keep
        min_fps=5,
        max_fps=60,
        init_after_n=10,             # wait until we have at least N deltas before initializing writers
    ):
        self.model = model

        # Store (base64_str, enqueue_time)
        self.to_process = []
        self.to_pose_output = []
        self.to_mask_output = []
        self.to_part_output = []
        self.to_mesh_output = []

        # ---- Video saving setup ----
        self.save_webcam_results = save_webcam_results
        self.save_original = bool(save_original)
        self.prefer_mp4 = bool(prefer_mp4)
        self.out_name_prefix = out_name_prefix

        # Desired / fallback fps
        self.video_fps = max(1, int(video_fps))

        # ---- input-fps matching ----
        self.match_input_fps = bool(match_input_fps)
        self.fps_window = int(fps_window)
        self.min_fps = float(min_fps)
        self.max_fps = float(max_fps)
        self.init_after_n = int(init_after_n)

        self._last_enqueue_t = None
        self._enqueue_dts = []      # rolling list of dt between enqueues (seconds)
        self.input_fps_est = None   # estimated input fps

        self.video_initialized = False
        self.save_root = os.path.join(os.getcwd(), "webcam-results")
        if self.save_webcam_results:
            os.makedirs(self.save_root, exist_ok=True)

        # VideoWriter handles
        self.pose_writer = None
        self.mask_writer = None
        self.part_writer = None
        self.mesh_writer = None
        self.orig_writer = None     # raw camera frames

        self.video_size = None      # (w, h) — set from model-output frame
        self.orig_size  = None      # (w, h) — set from raw camera frame

        # ---- pacing ----
        self.video_period = 1.0 / float(self.video_fps)
        self.next_write_time = None

        # Ensure files finalize even if program exits unexpectedly
        atexit.register(self.release_videos)

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    # ---------------------------------------------------
    # FPS estimation from input (enqueue timing)
    # ---------------------------------------------------
    def _update_input_fps_estimate(self, enqueue_t: float):
        if self._last_enqueue_t is None:
            self._last_enqueue_t = enqueue_t
            return

        dt = enqueue_t - self._last_enqueue_t
        self._last_enqueue_t = enqueue_t

        # Ignore ridiculous dt (e.g., pauses)
        if dt <= 1e-6 or dt > 5.0:
            return

        self._enqueue_dts.append(dt)
        if len(self._enqueue_dts) > self.fps_window:
            self._enqueue_dts.pop(0)

        if len(self._enqueue_dts) >= self.init_after_n:
            med_dt = float(np.median(self._enqueue_dts))
            if med_dt > 1e-6:
                fps = 1.0 / med_dt
                fps = float(np.clip(fps, self.min_fps, self.max_fps))
                self.input_fps_est = fps

    # ---------------------------------------------------
    # Core processing
    # ---------------------------------------------------
    def process_one(self):
        if not self.to_process:
            return

        str_image, enqueue_t = self.to_process.pop(0)

        # Update input-FPS estimate from enqueue times
        if self.match_input_fps:
            self._update_input_fps_estimate(enqueue_t)

        img = base64.b64decode(str_image)
        img_np = np.frombuffer(img, dtype="uint8")
        frame = cv2.imdecode(img_np, 1)         # raw camera frame (BGR)
        assert isinstance(frame, np.ndarray)

        # Model outputs (assumed BGR images)
        frame_pose, frame_mask, frame_part, frame_mesh = \
            self.model.get_analysis_results(frame, tracking=True)

        # ---- Initialize video writers lazily (after FPS known) ----
        if self.save_webcam_results and not self.video_initialized:
            if self.match_input_fps and self.input_fps_est is None:
                # Not enough data yet: do not init writers yet
                pass
            else:
                if self.match_input_fps and self.input_fps_est is not None:
                    self.video_fps = max(1, int(round(self.input_fps_est)))

                # Update pacing based on final fps used
                self.video_period = 1.0 / float(self.video_fps)

                # Pass raw frame so orig_writer can use its native resolution
                self.init_video_writers(frame, frame_pose, frame_mask, frame_part, frame_mesh)
                self.next_write_time = time()

        # ---- Write frames to video at that FPS (drop extras) ----
        if self.save_webcam_results and self.video_initialized:
            self.write_videos_paced(frame, frame_pose, frame_mask, frame_part, frame_mesh)

        # ---- Push clean frames to output queues ----
        self.to_pose_output.append(frame_pose)
        self.to_mask_output.append(frame_mask)
        self.to_part_output.append(frame_part)
        self.to_mesh_output.append(frame_mesh)

    # ---------------------------------------------------
    # Video handling
    # ---------------------------------------------------
    def _ensure_uint8_bgr(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def _open_writer(self, path: str, fourcc_str: str, fps: int, size_wh):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w, h = size_wh
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer is None or not writer.isOpened():
            return None
        return writer

    def init_video_writers(self, frame_orig, pose0, mask0, part0, mesh0):
        # ---- resolution for model-output writers ----
        pose0 = self._ensure_uint8_bgr(pose0)
        h, w = pose0.shape[:2]
        self.video_size = (w, h)

        # ---- resolution for original writer (native camera resolution) ----
        frame_orig = self._ensure_uint8_bgr(frame_orig)
        oh, ow = frame_orig.shape[:2]
        self.orig_size = (ow, oh)

        # ---- codec selection ----
        if self.prefer_mp4:
            candidates = [
                ("mp4v", "mp4"),
                ("avc1", "mp4"),
                ("H264", "mp4"),
                ("XVID", "avi"),
                ("MJPG", "avi"),
            ]
        else:
            candidates = [
                ("MJPG", "avi"),
                ("XVID", "avi"),
                ("mp4v", "mp4"),
            ]

        chosen = None
        for fourcc_str, ext in candidates:
            test_path = os.path.join(self.save_root, f"{self.out_name_prefix}_probe.{ext}")
            test_writer = self._open_writer(test_path, fourcc_str, self.video_fps, self.video_size)
            if test_writer is not None:
                test_writer.release()
                try:
                    os.remove(test_path)
                except OSError:
                    pass
                chosen = (fourcc_str, ext)
                break

        if chosen is None:
            print("[WARN] No working video encoder found. Videos will NOT be saved.")
            self.save_webcam_results = False
            self.video_initialized = False
            return

        fourcc_str, ext = chosen
        print(f"[INFO] Saving video at INPUT-FPS={self.video_fps} "
              f"codec={fourcc_str} ext=.{ext} "
              f"model-output size={w}x{h}  original size={ow}x{oh}")

        def _path(suffix):
            return os.path.join(self.save_root, f"{self.out_name_prefix}_{suffix}.{ext}")

        self.pose_writer = self._open_writer(_path("pose"), fourcc_str, self.video_fps, self.video_size)
        self.mask_writer = self._open_writer(_path("mask"), fourcc_str, self.video_fps, self.video_size)
        self.part_writer = self._open_writer(_path("part"), fourcc_str, self.video_fps, self.video_size)
        self.mesh_writer = self._open_writer(_path("mesh"), fourcc_str, self.video_fps, self.video_size)

        # Original video uses the camera's native resolution — never upscaled
        if self.save_original:
            self.orig_writer = self._open_writer(
                _path("original"), fourcc_str, self.video_fps, self.orig_size
            )
            if self.orig_writer is None:
                print("[WARN] Could not open original-frame writer. Raw video will NOT be saved.")
            else:
                print(f"[INFO] Original camera frames -> {_path('original')}")

        writers = [self.pose_writer, self.mask_writer, self.part_writer, self.mesh_writer]
        if any(wr is None for wr in writers):
            print("[WARN] Failed to open one or more writers. Disabling video saving.")
            self.release_videos()
            self.save_webcam_results = False
            self.video_initialized = False
            return

        self.video_initialized = True
        print(f"[INFO] Webcam video recording started -> {self.save_root}")

        # Write first frames immediately to ensure stream exists
        self.write_videos_once(frame_orig, pose0, mask0, part0, mesh0)

    def _resize_to(self, frame: np.ndarray, size_wh) -> np.ndarray:
        """Resize frame to (w, h) only if necessary."""
        if size_wh is None:
            return frame
        w, h = size_wh
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return frame

    def write_videos_once(self, frame_orig, frame_pose, frame_mask, frame_part, frame_mesh):
        """Write one frame to every active writer."""
        frame_pose = self._resize_to(self._ensure_uint8_bgr(frame_pose), self.video_size)
        frame_mask = self._resize_to(self._ensure_uint8_bgr(frame_mask), self.video_size)
        frame_part = self._resize_to(self._ensure_uint8_bgr(frame_part), self.video_size)
        frame_mesh = self._resize_to(self._ensure_uint8_bgr(frame_mesh), self.video_size)

        if any(f is None for f in [frame_pose, frame_mask, frame_part, frame_mesh]):
            return

        self.pose_writer.write(frame_pose)
        self.mask_writer.write(frame_mask)
        self.part_writer.write(frame_part)
        self.mesh_writer.write(frame_mesh)

        # Write raw camera frame at its own native resolution
        if self.save_original and self.orig_writer is not None:
            frame_orig = self._resize_to(self._ensure_uint8_bgr(frame_orig), self.orig_size)
            if frame_orig is not None:
                self.orig_writer.write(frame_orig)

    def write_videos_paced(self, frame_orig, frame_pose, frame_mask, frame_part, frame_mesh):
        """
        Write at real-time pace using self.video_fps (matched to input).
        Drops frames if processing is faster.
        """
        now = time()
        if self.next_write_time is None:
            self.next_write_time = now

        if now < self.next_write_time:
            return

        self.write_videos_once(frame_orig, frame_pose, frame_mask, frame_part, frame_mesh)

        self.next_write_time += self.video_period
        if self.next_write_time < now - self.video_period:
            self.next_write_time = now + self.video_period

    def release_videos(self):
        for wr in [self.pose_writer, self.mask_writer,
                   self.part_writer, self.mesh_writer,
                   self.orig_writer]:
            try:
                if wr is not None:
                    wr.release()
            except Exception:
                pass

        self.pose_writer = None
        self.mask_writer = None
        self.part_writer = None
        self.mesh_writer = None
        self.orig_writer = None
        self.video_initialized = False
        self.next_write_time = None

    # ---------------------------------------------------
    # Thread loop + public API
    # ---------------------------------------------------
    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.005)

    def enqueue_input(self, input):
        # Store enqueue time to measure INPUT fps
        self.to_process.append((input, time()))

    def get_pose_frame(self):
        while not self.to_pose_output:
            sleep(0.02)
        return self.to_pose_output.pop(0)

    def get_mask_frame(self):
        while not self.to_mask_output:
            sleep(0.02)
        return self.to_mask_output.pop(0)

    def get_part_frame(self):
        while not self.to_part_output:
            sleep(0.02)
        return self.to_part_output.pop(0)

    def get_mesh_frame(self):
        while not self.to_mesh_output:
            sleep(0.02)
        return self.to_mesh_output.pop(0)



# import threading
# import binascii
# from time import sleep, time
# import numpy as np
# import cv2
# import base64
#
#
# class Camera(object):
#     def __init__(self, model):
#         self.model = model
#         self.to_process = []
#         self.to_pose_output = []
#         self.to_mask_output = []
#         self.to_part_output = []
#         self.to_mesh_output = []
#         self.to_output = []
#         self.last_time = time()
#         self.fps = 0
#
#         thread = threading.Thread(target=self.keep_processing, args=())
#         thread.daemon = True
#         thread.start()
#
#     def process_one(self):
#         if not self.to_process:
#             return
#
#         str_image = self.to_process.pop(0)
#         img = base64.b64decode(str_image)
#         img_np = np.frombuffer(img, dtype='uint8')
#         frame = cv2.imdecode(img_np, 1)
#         assert type(frame) == np.ndarray
#
#         ret = self.model.get_analysis_results(frame, tracking=True)
#         frame_pose, frame_mask, frame_part, frame_mesh = ret
#
#         # Calculate FPS
#         current_time = time()
#         self.fps = int(1 / (current_time - self.last_time))
#         self.last_time = current_time
#
#         # Draw FPS on each frame
#         self.draw_fps(frame_pose)
#         self.draw_fps(frame_mask)
#         self.draw_fps(frame_part)
#         self.draw_fps(frame_mesh)
#
#         self.to_pose_output.append(frame_pose)
#         self.to_mask_output.append(frame_mask)
#         self.to_part_output.append(frame_part)
#         self.to_mesh_output.append(frame_mesh)
#
#     def draw_fps(self, frame):
#         """Draw FPS counter on the frame."""
#         cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#     def keep_processing(self):
#         while True:
#             self.process_one()
#             sleep(0.005)
#
#     def enqueue_input(self, input):
#         self.to_process.append(input)
#
#     def get_pose_frame(self):
#         while not self.to_pose_output:
#             sleep(0.02)
#         return self.to_pose_output.pop(0)
#
#     def get_mask_frame(self):
#         while not self.to_mask_output:
#             sleep(0.02)
#         return self.to_mask_output.pop(0)
#
#     def get_part_frame(self):
#         while not self.to_part_output:
#             sleep(0.02)
#         return self.to_part_output.pop(0)
#
#     def get_mesh_frame(self):
#         while not self.to_mesh_output:
#             sleep(0.02)
#         return self.to_mesh_output.pop(0)