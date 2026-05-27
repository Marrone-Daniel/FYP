from camera_controller import CameraController
import cv2

cam = CameraController(
    index_or_path=8,
    width=96,
    height=96,
    capture_width=640,
    capture_height=480,
    fps=30,
)

cam.start()
print("camera ready:", cam.is_ready)

while True:
    img = cam.get_image()
    print(img.shape, img.dtype, img.min(), img.max())

    # img 是 RGB，imshow 需要 BGR
    cv2.imshow("camera_8", img[..., ::-1])
    key = cv2.waitKey(30)
    if key == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()