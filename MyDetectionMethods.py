import cv2

class MyDetectionMethods:
    @staticmethod
    def canny_filter(image, lower, upper):
        """
        Applies Canny edge detection and finds contours.
        :param image: Input image (color image).
        :param lower_threshold: Input threshold decided by the user.
        :param upper_threshold: Input threshold decided by the user.
        :return: Tuple containing the image and the contours coordinates.

        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
      
        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, lower, upper)

        # Find contours from the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = image.copy()

        return result_image, contours

    @staticmethod
    def binarization(image, threshold):
        """
        Applies binarization and finds contours.
        :param image: Input image (color image).
        :param threshold: Input threshold decided by the user.
        :return: Tuple containing the image and the contours coordinates.

        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply inverse binary threshold
        _, binary_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours from the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = image.copy()

        return result_image, contours

    @staticmethod
    def adaptive_threshold(image, block_size, C):
        """
        Applies adaptive thresholding and finds contours.
        :param image: Input image (color image).
        :param block_size: Size of the neighborhood used for adaptive thresholding. Must be odd and greater than 1.
        :param C: Constant subtracted from the mean or weighted mean to fine-tune the thresholding.
        :return: Tuple containing the image and the contours coordinates.
        
        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply adaptive thresholding
        adaptive_thresh_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C
        )

        # Find contours from the thresholded image
        contours, _ = cv2.findContours(adaptive_thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = image.copy()

        return result_image, contours
    

