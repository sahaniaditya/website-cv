import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Show PLY as image
def show_ply_as_image(ply_path):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    # Create visualization window (offscreen)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()

    # Screenshot to numpy
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # Convert to displayable image
    img = (np.asarray(image) * 255).astype(np.uint8)
    return Image.fromarray(img)
