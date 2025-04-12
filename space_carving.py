import scipy.io
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import vtk

def run_space_carving():
    data = scipy.io.loadmat("uploads_spacecarving/data/dino_Ps.mat")["P"]
    projections = [data[0, i] for i in range(data.shape[1])]
    
    # === Load and preprocess images ===
    files = sorted(glob.glob("uploads_spacecarving/data/*.ppm"))
    images = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(float)
        im /= 255
        images.append(im[:, :, ::-1])  # BGR to RGB
    
    # === Create silhouettes ===
    imgH, imgW, _ = images[0].shape
    silhouettes = []
    for im in images:
        mask = np.abs(im - [0.0, 0.0, 0.75])
        mask = np.sum(mask, axis=2)
        y, x = np.where(mask <= 1.1)
        im[y, x, :] = [0.0, 0.0, 0.0]
        im = im[:, :, 0]
        im[im > 0] = 1.0
        im = im.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
        silhouettes.append(im)
    
    # === Create voxel grid ===
    s = 120
    x, y, z = np.mgrid[:s, :s, :s]
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float).T
    nb_points_init = pts.shape[0]
    
    # Normalize and center
    pts[:, 0] /= np.max(pts[:, 0])
    pts[:, 1] /= np.max(pts[:, 1])
    pts[:, 2] /= np.max(pts[:, 2])
    center = np.mean(pts, axis=0)
    pts -= center
    pts /= 5
    pts[:, 2] -= 0.62
    
    # Homogeneous coordinates
    pts_hom = np.vstack((pts.T, np.ones((1, nb_points_init))))
    
    # === Voxel carving: count silhouettes where voxel is occupied ===
    filled = []
    for P, sil in zip(projections, silhouettes):
        uvs = P @ pts_hom
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        x_valid = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < imgW)
        y_valid = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < imgH)
        valid = np.logical_and(x_valid, y_valid)
        indices = np.where(valid)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        res = sil[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res
        filled.append(fill)
    
    filled = np.vstack(filled)
    occupancy = np.sum(filled, axis=0)
    
    # === Save voxel grid as .vtr (only the voxels with occupancy > threshold) ===
    threshold = 25
    occupancy_mask = (occupancy > threshold).astype(np.float32)
    
    # Create grid coordinates
    x_coords = sorted(list(set(np.round(pts[:, 0][::s*s], 6))))
    y_coords = sorted(list(set(np.round(pts[:, 1][:s*s:s], 6))))
    z_coords = sorted(list(set(np.round(pts[:, 2][:s], 6))))
    
    x_array = vtk.vtkFloatArray()
    y_array = vtk.vtkFloatArray()
    z_array = vtk.vtkFloatArray()
    
    for val in x_coords:
        x_array.InsertNextValue(val)
    for val in y_coords:
        y_array.InsertNextValue(val)
    for val in z_coords:
        z_array.InsertNextValue(val)
    
    # Only add occupancy values for retained voxels
    values = vtk.vtkFloatArray()
    values.SetName("Occupancy")
    for i in range(len(occupancy_mask)):
        values.InsertNextValue(occupancy_mask[i])
    
    # Create rectilinear grid
    rgrid = vtk.vtkRectilinearGrid()
    rgrid.SetDimensions(len(x_coords), len(y_coords), len(z_coords))
    rgrid.SetXCoordinates(x_array)
    rgrid.SetYCoordinates(y_array)
    rgrid.SetZCoordinates(z_array)
    rgrid.GetPointData().SetScalars(values)
    
    # Save to .vtr
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName("res_space/shape.vtr")
    writer.SetInputData(rgrid)
    writer.Write()
        