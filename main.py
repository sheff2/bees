import cv2

input_video = "secondbees.mp4"

# video Inference
def vid_inf(vid_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(vid_path)
    # Get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = "output_recorded.mp4"

    # Create the `VideoWriter()` object
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Create Background Subtractor MOG2 object
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Apply background subtraction
            fg_mask = backSub.apply(frame)

            # Apply global threshold to remove shadows
            retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

            # Set the kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Apply erosion
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_contour_area = 500  # Define your minimum area threshold
            # Filter contours based on area
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            # Count the number of detected objects in the current frame
            object_count = len(large_contours)

            frame_out = frame.copy()

            # Draw bounding boxes around detected objects
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 3)

            # Add text box in the top right corner displaying the object count
            text = f"Objects Detected: {object_count}"
            font = cv2.FONT_HERSHEY_PLAIN
            text_size, _ = cv2.getTextSize(text, font, 1, 2)
            text_x = frame_width - text_size[0] - 10  # 10 pixels from right edge
            text_y = 30  # Position 30 pixels from top

            # Draw a rectangle behind the text for visibility
            cv2.rectangle(frame_out, (text_x - 5, text_y - 25), (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
            # Overlay the text
            cv2.putText(frame_out, text, (text_x, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Saving the video file
            out.write(frame_out)

            # Display the resulting frame
            cv2.imshow("Frame_final", frame_out)

            # Press Q on keyboard to exit
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            break

    # When everything done, release the video capture and writer object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()

vid_inf(input_video)
