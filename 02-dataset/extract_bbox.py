import cv2
import argparse

points = []
bbox = None
image = None
temp_image = None

def mouse_callback(event, x, y, flags, param):
    global points, bbox, image, temp_image

    # if left mouse button is clicked, store the point and draw a circle
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(temp_image, (x, y), 4, (0, 255, 0), -1)

        # if two points are clicked
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]

            # get the bounding box coordinates
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)

            # normalize the bounding box coordinates
            img_h, img_w = image.shape[:2]

            # calculate the bounding box in the right format
            bbox = [
                x_min / img_w,
                y_min / img_h,
                (x_max - x_min) / img_w,
                (y_max - y_min) / img_h,
            ]

            # draw the bounding box on the image
            cv2.rectangle(temp_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            print("[")
            
            # print the bounding box values with 8 decimal places to match the format
            for value in bbox:
                print(f"    {value:.8f},")
            print("]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click 2 points to extract a bounding box")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    image = cv2.imread(args.input_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input_path}")

    temp_image = image.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    while True:
        cv2.imshow("image", temp_image)
        key = cv2.waitKey(1)

        # if "r" reset
        if key == ord("r") or key == 27:
            temp_image = image.copy()
            points = []
            bbox = None

        # if "q" or "enter" is pressed, break the loop
        if key == ord("q") or key == 13:
            break

    cv2.destroyAllWindows()

    if bbox is not None:
        print("final bbox:", bbox)