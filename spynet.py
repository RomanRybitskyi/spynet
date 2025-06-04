import cv2
import numpy as np
import torch
from model import estimate
from flowiz import convert_from_flow

def live_spynet_visualization(source=0, save_output=False, output_path="live_spynet_output.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    prev_frame = None
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (1024, 416))

        if prev_frame is not None:
            tenOne = torch.FloatTensor(np.ascontiguousarray(prev_frame[:, :, ::-1].transpose(2, 0, 1)) / 255.0).to(device)
            tenTwo = torch.FloatTensor(np.ascontiguousarray(frame_resized[:, :, ::-1].transpose(2, 0, 1)) / 255.0).to(device)

            try:
                flow = estimate(tenOne, tenTwo).cpu().numpy().transpose(1, 2, 0)
                flow_img = convert_from_flow(flow)
            except Exception as e:
                print(f"[!] Flow estimation failed: {e}")
                continue
            # Convert input frame for display (RGB to BGR for OpenCV)
            input_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            flow_bgr = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

            # Concatenate input and flow image horizontally
            combined = np.hstack((input_bgr, flow_bgr))
            # Show result
            cv2.imshow("Input vs Optical Flow", combined)
            # cv2.imshow("SPyNet Optical Flow", cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))
    
            if save_output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, 10.0, (1024, 416))
                writer.write(cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))

        prev_frame = frame_resized

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # live_spynet_visualization(0)  # вебкамера
    live_spynet_visualization("videos/1080.mp4", save_output=True, output_path="output_videos/live_spynet_output_1080.mp4")  # відеофайл
    # live_spynet_visualization("http://192.168.X.X:8080/video", save_output=True, output_path="spynet_live_saved.mp4")
