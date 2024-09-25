# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt

# model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
# Use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model == "DPT_Large" or model == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to("cpu")

    with torch.no_grad():
        prediction = model(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        # Find the center pixel
        H, W = output.shape
        center_x, center_y = W // 2, H // 2
        center_depth = output[center_y, center_x]  # Extract depth at the center pixel
        center_depth_in_meters = center_depth
        # Draw a rectangle around the center pixel
        box_size = 50  # Size of the box
        top_left = (center_x - box_size // 2, center_y - box_size // 2)
        bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green box

        # Display the depth below the box
        text_position = (
            center_x - box_size // 2,
            center_y + box_size // 2 + 30,
        )
        cv2.putText(
            frame,
            f"Depth: {center_depth_in_meters:.2f} m",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    plt.imshow(output)
    cv2.imshow("CV2Frame", frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()

plt.show()
