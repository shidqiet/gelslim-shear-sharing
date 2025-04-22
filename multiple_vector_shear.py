import time

import cv2
import torch

from gelslim_shear.plot_utils.shear_plotter import ShearPlotter
from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator

# Video Input
video_input = cv2.VideoCapture("data/people.mp4")
ret, first_frame_np = video_input.read()
H, W, _ = first_frame_np.shape
base_tactile_image = torch.from_numpy(first_frame_np).permute(2, 0, 1).float()

# Initialize ShearGenerator and ShearPlotter
shgen = ShearGenerator(
    method="weighted",
    channels=[
        "u",
        "v",
        "div",
        "curl",
        "sol_u",
        "sol_v",
        "irr_u",
        "irr_v",
        "dudt",
        "dvdt",
        "du",
        "dv",
    ],
    Farneback_params=(0.5, 3, 45, 3, 5, 1.2, 0),
)
shgen.reset_shear(base_tactile_image)
shplot = ShearPlotter(
    channels=shgen.channels,
    titles=[
        "Shear Field",
        "Solenoidal",
        "Irrotational",
        "Time Derivative",
        "Change",
        "Divergence",
        "Curl",
    ],
)


# Define the update function for animation
def update(frame):
    ret, frame_np = video_input.read()
    if not ret:
        return shplot.plots  # Stop the animation if no frame is read

    # Convert the frame to a tensor (similar to how base_tactile_image was processed)
    tactile_image = torch.from_numpy(frame_np).permute(2, 0, 1).float()

    # Update shear generator with the new tactile image and time
    t = time.time()
    shgen.update_time(t)
    shgen.update_tactile_image(tactile_image)
    shgen.update_shear()
    shear_field_tensor = shgen.get_shear_field()

    # Update the shear plotter with the new shear field data
    shplot.update_shear_info(frame, [shear_field_tensor])
    return shplot.plots


tactile_image = base_tactile_image.clone()
shgen.update_time(time.time())
shgen.update_tactile_image(tactile_image)
shgen.update_shear()
shear_field_tensor = shgen.get_shear_field()

# Create the animation (frames will be updated each time the update function is called)
ani = shplot.animate_shear_info(
    [shear_field_tensor],
    update,
)

# Release the video capture after the animation is saved
video_input.release()
