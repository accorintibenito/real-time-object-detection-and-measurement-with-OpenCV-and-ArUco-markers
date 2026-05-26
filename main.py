import cv2
import cv2.aruco as aruco
import numpy as np
from MyDetectionMethods import MyDetectionMethods 

""" 
For the demostration you will see only the Adaptive thresholding in the video since it gives the best results.

Go to the end of the code to see comments about how the image "travels" through the program.
"""

def calculate_pixel_to_cm_ratio(corners, aruco_size_cm):
    """
    Calculate the pixel-to-cm ratio using the ArUco marker.
    :param corners: Detected corners of the ArUco marker, detected further on in the code, it is a
                    2D array containing the pixel coordinates of the four corners of the marker.
    :param aruco_size_cm: Real-world size of the ArUco marker in cm, used as a reference.
    :return pixel-to-cm ratio: to calculate it, I also performed an average of the marker's width and heigt in pixels,
                                in order to  mitigates effects such distortions, or non ideal camera angle 
                                (ideal: perpendicular to the marker)
    """
    if corners is not None:
        # Extract the coordinates of the four corners of the ArUco marker
        top_left, top_right, bottom_right, bottom_left = corners[0][0]
        # Calculate the Euclidean distance (in pixels) between the top_left and top_right corners, which is the marker's width in the image
        # top_right - top_left is a difference between numpy arrays that have form [x,y], 
        # so are performed the following differences (x_top_right - x_top_left, y_top_right - y_top_left) for example
        # then np.linalg.norm calculates the euclidian distance of difference vector given,
        # so this calculation is performed ==> sqrt((x_right - x_left)^2 + (y_right - y_left)^2)
        width = np.linalg.norm(top_right - top_left)
        # Calculate the Euclidean distance (in pixels) between the top_left and bottom_left corners, which is the marker's height in the image. 
        # Same explanation of operations performed by np.linalg.norm
        height = np.linalg.norm(top_left - bottom_left)
        # Compute the pixel to cm ratio
        return (width + height) / 2 / aruco_size_cm
    return None

def main():

    # Open the external camera (ID 1)
    cap = cv2.VideoCapture(1)  
    # If the the camera does not open or it doesn't find it, then the program returns an error
    if not cap.isOpened():
        print("Error: Impossible to open the camera.")
        return

    # Prompt the user to choose a filtering method
    print("Choose the filtering method:")
    print("1. Canny Filter")
    print("2. Binarization")
    print("3. Adaptive Thresholding")
    choice = input("Insert the number of your request: ")

    # Initialize lower and upper threshold for the canny and the threshold for the binarization
    lower_threshold, upper_threshold, bin_threshold = None, None, None

    if choice == "1":
        lower_threshold = int(input("Insert the lower threshold for the Canny filter: "))
        upper_threshold = int(input("Insert the upper threshold for the Canny filter: "))
    elif choice == "2":
        bin_threshold = int(input("Insert the threshold for the Binarization filter: "))
    elif choice == "3":
        block_size=11
        C=2
    else:
        print("Choice not valid. Restart the program and select 1, 2 or 3.")
        return

    # Load ArUco dictionary to detect ArUco markers in the frame
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    
    # Set the real-world size of the ArUco marker in cm
    aruco_size_cm = 10.0  

    # Until we don't press 'q' or close the window by ourself we stay inside the while
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # If the frame is not read, then it returns an error
        if not ret:
            print("Error: impossible to read the frame.")
            break

        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect any ArUco markers in the gray scale frame 
        # and extract their corners and IDs
        # corners: a list containing the coordinates of the four corners for each detected ArUco marker in the image.
        # ids: a numpy array containing the unique IDs of the detected markers
        corners, ids = detector.detectMarkers(gray)

        # Initialize the pixel to cm ratio
        pixel_to_cm_ratio = None

        # If there are IDs in the numpy array "ids" containing the unique IDs of the detected markers then we
        # - draw respective corners and IDs of the markers on the frame 
        # - calculate the pixel-to-cm ratio and print it on the top left corner of the frame
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            pixel_to_cm_ratio = calculate_pixel_to_cm_ratio(corners, aruco_size_cm)
            print(f"Pixel-to-CM Ratio: {pixel_to_cm_ratio:.2f}")
            cv2.putText(frame, f"Pixel-to-CM Ratio: {pixel_to_cm_ratio:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # After we have calculated the pixel-to-cm ratio
        if pixel_to_cm_ratio is not None:
            
            # if we have chosen 1 then we apply the canny filter
            if choice == "1":
                # Extract the image filtered with the canny filter and the contours
                result_canny, contours_canny = MyDetectionMethods.canny_filter(frame, lower_threshold, upper_threshold)

                for contour in contours_canny:
                    
                    # Calculates a bounding rectangle that encloses the given contour
                    # The function returns:
                    # - x: The x-coordinate of the top-left corner of the rectangle
                    # - y: The y-coordinate of the top-left corner of the rectangle
                    # - w: The width of the rectangle in pixels
                    # - h: The height of the rectangle in pixels
                    x, y, w, h = cv2.boundingRect(contour)

                    # Converts the width 'w' and height 'h' of the bounding rectangle 
                    # from pixels to centimeters using the previously calculated pixel_to_cm_ratio
                    length_cm = w / pixel_to_cm_ratio
                    height_cm = h / pixel_to_cm_ratio

                    # Check for Credit Card sizes (lenght and height) with a +/- 5mm precision
                    # Draw a green bounding rectangle on the frame to show that the object is detected
                    # Print in blue Credit Card sizes starting from the top left corner of the bounding rectangle
                    if (8.5 <= length_cm <= 9.5 and 5.0 <= height_cm <= 6.0):  # Credit Card
                        cv2.rectangle(result_canny, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_canny, f"Credit Card L: {length_cm:.2f} cm, H: {height_cm:.2f} cm", 
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Check for Battery AAA sizes (lenght and height) with a +/- 5mm precision
                    # Draw a green bounding rectangle on the frame to show that the object is detected
                    # Print in red Battery AAA sizes starting from the top left corner of the bounding rectangle
                    elif (4.0 <= length_cm <= 5.0 and 0.5 <= height_cm <= 1.5):  # Battery AAA
                        cv2.rectangle(result_canny, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_canny, f"Battery L: {length_cm:.2f} cm, H: {height_cm:.2f} cm", 
                                    (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                # Show the Canny filter window with 
                # - ArUco markers corners and IDs
                # - bounding rectangles of specific objects
                # - prints of pixel to cm ratio and specific object sizes
                cv2.imshow("Canny Filter", result_canny)

            # if we have chosen 2 then we apply the binarization 
            elif choice == "2":
                # Extract the image filtered with the binarization filter and the contours
                result_binarization, contours_binarization = MyDetectionMethods.binarization(frame, bin_threshold)

                for contour in contours_binarization:
                    
                    # Calculates a bounding rectangle that encloses the given contour
                    # The function returns:
                    # - x: The x-coordinate of the top-left corner of the rectangle
                    # - y: The y-coordinate of the top-left corner of the rectangle
                    # - w: The width of the rectangle in pixels
                    # - h: The height of the rectangle in pixels
                    x, y, w, h = cv2.boundingRect(contour)
                    length_cm = w / pixel_to_cm_ratio
                    height_cm = h / pixel_to_cm_ratio

                    # Check for Credit Card sizes (lenght and height) with a +/- 5mm precision
                    # Draw a green bounding rectangle on the frame to show that the object is detected
                    # Print in blue Credit Card sizes starting from the top left corner of the bounding rectangle
                    if (8.5 <= length_cm <= 9.5 and 5.0 <= height_cm <= 6.0):  # Credit Card
                        cv2.rectangle(result_binarization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_binarization, f"Credit Card L: {length_cm:.2f} cm, H: {height_cm:.2f} cm", 
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Check for Battery AAA sizes (lenght and height) with a +/- 5mm precision
                    # Draw a green bounding rectangle on the frame to show that the object is detected
                    # Print in red Battery AAA sizes starting from the top left corner of the bounding rectangle                     
                    elif (4.0 <= length_cm <= 5.0 and 0.5 <= height_cm <= 1.5):  # Battery AAA
                        cv2.rectangle(result_binarization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_binarization, f"Battery L: {length_cm:.2f} cm, H: {height_cm:.2f} cm", 
                                    (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                # Show the Binarization window with 
                # - ArUco markers corners and IDs
                # - bounding rectangles of specific objects
                # - prints of pixel to cm ratio and specific object sizes
                cv2.imshow("Binarization", result_binarization)

            # if we have chosen 3 then we apply the adaptive thresholding                
            elif choice == "3":
                # Extract the image filtered with the adaptive thresholding and the contours
                result_adaptive, contours_adaptive = MyDetectionMethods.adaptive_threshold(frame, block_size, C)
                
                for contour in contours_adaptive:
                    
                    # Calculates a bounding rectangle that encloses the given contour
                    # The function returns:
                    # - x: The x-coordinate of the top-left corner of the rectangle
                    # - y: The y-coordinate of the top-left corner of the rectangle
                    # - w: The width of the rectangle in pixels
                    # - h: The height of the rectangle in pixels
                    x, y, w, h = cv2.boundingRect(contour)
                    length_cm = w / pixel_to_cm_ratio
                    height_cm = h / pixel_to_cm_ratio

                    # Check for Credit Card sizes (lenght and height) with a +/- 5mm precision
                    # Draw a green bounding rectangle on the frame to show that the object is detected
                    # Print in blue Credit Card sizes starting from the top left corner of the bounding rectangle
                    if (8.5 <= length_cm <= 9.5 and 5.0 <= height_cm <= 6.0):  # Credit Card
                        cv2.rectangle(result_adaptive, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_adaptive, f"Credit Card L: {length_cm:.2f} cm, H: {height_cm:.2f} cm", 
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                    # Check for Battery AAA sizes (lenght and height) with a +/- 5mm precision
                    # Draw a green bounding rectangle on the frame to show that the object is detected
                    # Print in red Battery AAA sizes starting from the top left corner of the bounding rectangle
                    elif (4.0 <= length_cm <= 5.0 and 0.5 <= height_cm <= 1.5):  # Battery AAA
                        cv2.rectangle(result_adaptive, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_adaptive, f"Battery L: {length_cm:.2f} cm, H: {height_cm:.2f} cm", 
                                    (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Show the Adaptive Thresholding window with 
                # - ArUco markers corners and IDs
                # - bounding rectangles of specific objects
                # - prints of pixel to cm ratio and specific object sizes                        
                cv2.imshow("Adaptive Thresholding", result_adaptive)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all the open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

""" 
### **How the Image Travels Through the Program:**
1. Capture Frame:
   - A frame is captured from the external camera using OpenCV.
   - The frame is converted to grayscale for processing.

2. ArUco Marker Detection:
   - The grayscale frame is analyzed for ArUco markers.
   - If a marker is detected, its corners are used to calculate the pixel-to-cm ratio.

3. Preprocessing with Selected Filter:
   - Depending on the user's choice, the frame undergoes:
     - **Canny Filter:** Detects edges by calculating intensity gradients, look inside MyDetectionMethods.py for the preprocessing steps.
     - **Binarization:** Converts the frame into a binary image based on a user-defined threshold, look inside MyDetectionMethods.py for the preprocessing steps.
     - **Adaptive Thresholding:** Segments the frame using local pixel intensity variations, look inside MyDetectionMethods.py for the preprocessing steps.

4. Contour Detection:
   - Contours are extracted from the processed image using OpenCV's `findContours()` function.

5. Object Measurement and Classification:
   - Each contour is enclosed in a bounding rectangle.
   - The rectangle's dimensions (width and height in pixels) are converted to real-world measurements (in cm) using the pixel-to-cm ratio.
   - The dimensions are compared to predefined ranges to classify objects as credit cards or AAA batteries.

6. Display Results:
   - The processed frame is displayed with the following:
     - ArUco marker outlines and IDs.
     - Bounding rectangles and labels for identified objects.
     - Measured pixel-to-cm ratio.

### **Final Output:**
- A real-time video with annotated objects and their measurements.
- Interactive selection of filtering methods and object detection with easy exit functionality in the program console.

"""
