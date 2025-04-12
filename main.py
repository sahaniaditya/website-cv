import streamlit as st
import os
import shutil
import zipfile
from space_carving import run_space_carving
from sfm import run_sfm
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D Reconstruction Web App", layout="wide")
st.markdown("<div style='text-align: center;'><h1>3D Scene Reconstruction</h1></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'><h2>Computer Vision Course Project</h2></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: right; font-weight:none;color : gray;'><h5>Course Instructor - Dr. Pratik Mazumdar</h5></div>", unsafe_allow_html=True)
st.markdown(
    "<h4><a href='https://github.com/majisouvik26/3d-scene-reconstruction' target='_blank'>Github Link of Project</a></h4>",
    unsafe_allow_html=True
)
st.header("Team Member:")
st.markdown("<div> <ul><li>Aditya Sahani(B22CS003)</li><li>Veeraraju Elluru(B22CS080)</li><li>Souvik Maji(B22CS089)</li><li>Dishit Sharma(B22CS082)</li><li>Aditya Rathor(B22AI044)</li></ul></div>", unsafe_allow_html=True)


st.header("Introduction")
st.write("3D scene reconstruction is a fundamental problem in computer vision that involves recovering the three-dimensional structure of a scene from a set of two-dimensional images. " \
"The goal is to generate an accurate and detailed representation of the real world, typically in the form of point clouds, meshes, or volumetric models. This process plays a crucial role in various applications such as robotics, augmented and virtual reality (AR/VR), autonomous navigation, and cultural heritage preservation." \
" The reconstruction pipeline often incorporates techniques like multi-view stereo, Structure from Motion (SfM), and volumetric methods like voxel carving. By leveraging image geometry, camera calibration data, and feature correspondences across views, 3D reconstruction enables machines to perceive and interpret the spatial layout of the physical world.")


st.image("https://filelist.tudelft.nl/BK/Onderzoek/Research_stories/zhaiyu.png", caption="3D Reconstruction")


st.header("Methods Used and Results")
st.subheader("1. NeRF - Neural Radiance Fields (NeRF)")

col1, col2 = st.columns(2)

with col1:
    st.image("media/test_image_0.png", width=400, caption="Input Image")

with col2:
    st.image("media/truck_reconstruction.gif", width=400, caption="3D Reconstruction")

st.subheader("2. Space Carving")
col1, col2 = st.columns(2)

with col1:
    st.image("media/input_images.png", width=400, caption="Input Image")

with col2:
    st.image("media/shape_mesh.png", width=400, caption="3D Reconstruction")

st.subheader("3. Pix2Vox")
col1, col2 = st.columns(2)

with col1:
    st.image("media/pix.jpg", width=400, caption="Input Image")

with col2:
    st.image("media/pix_output.jpg", width=400, caption="3D Reconstruction")


st.subheader("4. SFM Method")
col1, col2 = st.columns(2)

with col1:
    st.image("media/DSC_0351.JPG", width=400, caption="Input Image")

with col2:
    st.image("media/image.png", width=400, caption="3D Reconstruction")

st.subheader("5. Incremental SFM Method")
col1, col2 = st.columns(2)

with col1:
    st.image("media/WhatsApp Image 2025-04-12 at 17.40.27_1137ddf7.jpg", width=400, caption="Input Image")

with col2:
    st.image("media/rotation_sfm_cam_1_2_3_4[1].gif", width=400, caption="3D Reconstruction") 

st.subheader("6. Gaussian Splatting Method")
# col1, col2 = st.columns(2)

# with col1:
#     st.image("WhatsApp Image 2025-04-12 at 17.40.27_1137ddf7.jpg", width=400, caption="Input Image")

# with col2:
st.image("media/gs.gif", width=400, caption="3D Reconstruction")         




st.header("DEMO OF MODELS")

def show_ply_interactive(ply_path):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # Optional: use colors
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.full_like(points, fill_value=0.5)  # default gray

    # Create interactive plot
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=colors,
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
        ),
        width=800,
        height=600,
        margin=dict(r=10, l=10, b=10, t=10)
    )

    return fig

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

# ---------- Function to extract zip ----------
def extract_zip(zip_file, extract_to):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to

# ---------- SPACE CARVING ----------
st.header("📦 Space Carving")
st.markdown("""
**Space Carving** is a volumetric method that uses silhouettes from multiple views to reconstruct a 3D object by carving away inconsistent voxels.

👉 Upload a `.zip` file containing images (JPG/PNG) from different calibrated views.
""")

sc_zip = st.file_uploader("Upload ZIP file for Space Carving", type=["zip"])

if sc_zip:
    sc_extract_path = "uploads_spacecarving"
    with open("temp_spacecarving.zip", "wb") as f:
        f.write(sc_zip.getbuffer())

    extract_zip("temp_spacecarving.zip", sc_extract_path)
    st.success("Extracted  images.")

    if st.button("Run Space Carving Model"):
        output = run_space_carving()  # This should generate the .vtr file
        st.success("Model ran successfully.")
    
        # Path to generated .vtr file
        vtr_path = "res_space/shape.vtr"  # Update if filename differs
    
        if os.path.exists(vtr_path):
            st.markdown("### 📥 Download Space Carved VTR File")
            with open(vtr_path, "rb") as f:
                st.download_button(
                    label="Download .vtr file",
                    data=f,
                    file_name=os.path.basename(vtr_path),
                    mime="application/octet-stream"
                )
        else:
            st.warning("No .vtr file found. Make sure the model ran successfully.")


# ---------- STRUCTURE FROM MOTION ----------
st.markdown("---")
st.header("📷 Structure from Motion (SfM)")
st.markdown("""
**Structure from Motion (SfM)** reconstructs 3D geometry and camera poses from a series of images.

👉 Upload a `.zip` file containing your image dataset (JPG/PNG).
""")

sfm_zip_file = st.file_uploader("Upload ZIP file for SfM", type=["zip"])

if sfm_zip_file is not None:
    zip_name = os.path.splitext(sfm_zip_file.name)[0]  # 👉 'dataset.zip' → 'dataset'
    sfm_extract_path = "uploads_sfm"

    extract_zip(sfm_zip_file, sfm_extract_path)
    st.success(f"Extracted {zip_name} dataset.")

    if st.button("Run SfM Model"):
        output = run_sfm(sfm_extract_path + "\\" + zip_name) 
        st.success("Model ran successfully.")

        # Construct PLY path based on zip filename
        ply_path = os.path.join("res", f"{zip_name}.ply")

        if os.path.exists(ply_path):
            st.markdown("### 🧩 Reconstructed Point Cloud Image")
            image = show_ply_as_image(ply_path)
            st.image(image, caption=f"{zip_name}.ply", use_column_width=True)

            # Optional download
            with open(ply_path, "rb") as f:
                st.download_button(
                    label="📥 Download .ply file",
                    data=f,
                    file_name=f"{zip_name}.ply",
                    mime="application/octet-stream"
                )
        else:
            st.warning(f"No .ply file named {zip_name}.ply found in 'res/'.")

        if os.path.exists(ply_path):
            st.markdown("### 🧩 Reconstructed Point Cloud (Interactive)")
            fig = show_ply_interactive(ply_path)
            st.plotly_chart(fig, use_container_width=True)    
