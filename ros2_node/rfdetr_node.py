import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage
import supervision as sv
from rfdetr import RFDETRSmall

# Path to your fine-tuned checkpoint (EMA weights generalize best)
MODEL_WEIGHTS = "output/checkpoint_best_ema.pth"

# Class id -> name mapping from YOUR training dataset
CLASSES = {
    0: "forklift",
    1: "person",
}

CONFIDENCE_THRESHOLD = 0.4  # slightly lower than 0.5: steadier detections help the tracker


class RFDETRNode(Node):
    def __init__(self):
        super().__init__("rfdetr_node")
        self.bridge = CvBridge()

        self.model = RFDETRSmall(pretrain_weights=MODEL_WEIGHTS)
        self.model.optimize_for_inference()

        # ByteTrack assigns a persistent id to each object across frames
        self.tracker = sv.ByteTrack(frame_rate=30)

        self.trace_annotator = sv.TraceAnnotator()      # motion trails
        self.box_annotator = sv.BoxAnnotator(thickness=4)
        self.label_annotator = sv.LabelAnnotator()

        self.sub = self.create_subscription(
            Image, "/warehouse_camera/rgb", self.on_image, 1)
        self.pub_annotated = self.create_publisher(Image, "/rfdetr/annotated", 1)
        self.pub_detections = self.create_publisher(Detection2DArray, "/rfdetr/detections", 1)
        self.get_logger().info(
            f"RF-DETR tracking node ready (weights: {MODEL_WEIGHTS}), waiting for images...")

    def class_name(self, cid) -> str:
        # Fallback to the raw id so a wrong mapping shows up as numbers, not a crash
        return CLASSES.get(int(cid), str(int(cid)))

    def on_image(self, msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = self.model.predict(
            PILImage.fromarray(frame_rgb), threshold=CONFIDENCE_THRESHOLD)

        # Associate detections across frames -> persistent tracker_id per object
        detections = self.tracker.update_with_detections(detections)

        # --- annotated image for humans (RViz) ---
        labels = [f"#{tid} {self.class_name(cid)} {conf:.2f}"
                  for cid, conf, tid in zip(
                      detections.class_id, detections.confidence, detections.tracker_id)]
        annotated = self.trace_annotator.annotate(frame_bgr.copy(), detections)
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)
        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.pub_annotated.publish(out)

        # --- structured detections for machines (downstream nodes) ---
        arr = Detection2DArray()
        arr.header = msg.header
        for (x1, y1, x2, y2), cid, conf, tid in zip(
                detections.xyxy, detections.class_id,
                detections.confidence, detections.tracker_id):
            d = Detection2D()
            d.id = str(int(tid))  # persistent track id, e.g. "3"
            d.bbox.center.position.x = float((x1 + x2) / 2)
            d.bbox.center.position.y = float((y1 + y2) / 2)
            d.bbox.size_x = float(x2 - x1)
            d.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.class_name(cid)
            hyp.hypothesis.score = float(conf)
            d.results.append(hyp)
            arr.detections.append(d)
        self.pub_detections.publish(arr)


def main():
    rclpy.init()
    node = RFDETRNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()