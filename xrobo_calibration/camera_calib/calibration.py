from .intrinsic import calibrate_camera_intrinsic
from .extrinsic import calibrate_camera_extrinsic
from .pattern_detection import get_image_points

def calibrate_camera(images, pattern_size, verbose=True):
    """Calibrate camera using a set of images.
    
    Args:
        images (list): A list of images.
        pattern_size (dict): A dictionary containing the number of rows and columns of the pattern.
        verbose (bool): Whether to print the calibration results.
    
    Returns:
        tuple: A tuple containing the intrinsic and extrinsic calibration results.
    """
    
    # Detect image points
    image_points = get_image_points(images, pattern_size)
    
    # Define image size (assume all images have the same resolution)
    image_size = (images[0].shape[1], images[0].shape[0])
    
    # Do calibration
    intrinsic_results = calibrate_camera_intrinsic(image_points, image_size, pattern_size,
                                                   verbose=verbose)
    
    extrinsic_results = calibrate_camera_extrinsic(
                            image_points=image_points,
                            image_size=image_size,
                            pattern_size=pattern_size,
                            intrinsic_matrix=intrinsic_results['intrinsic_matrix'],
                            distortion_coeffs=intrinsic_results['distortion_coeffs'])

    if verbose:
        print(intrinsic_results)
        print(extrinsic_results)
        
    return intrinsic_results, extrinsic_results