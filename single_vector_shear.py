import time

import cv2
import torch

from gelslim_shear.plot_utils.shear_plotter import ShearPlotter
from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator

# Video Input
video_input = cv2.VideoCapture("data/people.mp4")
ret, first_frame = video_input.read()
H, W, _ = first_frame.shape
base_tactile_image = torch.from_numpy(first_frame).permute(2, 0, 1).float()

# Shear Generator
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

ret, frame = video_input.read()
tactile_image = torch.from_numpy(frame).permute(2, 0, 1).float()

t = time.time()
shgen.update_time(t)
shgen.update_tactile_image(tactile_image)
shgen.update_shear()
shear_field_tensor = shgen.get_shear_field()
shplot.plot_shear_info([shear_field_tensor])
shplot.show()
