# Voxel
#voxel
Think of a **voxel** (*vo*lumetric pi*xel*) as a 3D building block in your image dataset, just like a 2D digital image is a grid of square pixels, a 3D stack of microscope images is a grid of 3D voxels.

In a 2D camera, a pixel represents a square area (width $\times$ height). In a 3D microscope or CT scan, a voxel represents a **rectangular box** with volume (width $\times$ height $\times$ depth). Where:
- **$x, y$ coordinates** correspond to pixel locations within a single focal plane (a 2D image slice).
- **$z$ coordinate** corresponds to the optical section or slice index as the microscope moves vertically.
The following figure shows an example of a voxel. Anisotropy in voxels occurs when they are stretched along one axis (usually z) compared to the xy plane. 

![Voxel example](Excalidraw/voxel_example)

In the competition they scale the voxel values to represent **physical dimensions**, this is, correct for the distortion introduced by the measurement (optical setup of the microscope).
- **In-plane resolution ($x = y = 0.40625\ \mu\text{m}$):**
    
    The microscope's camera sensor and lens magnification resolve details down to $0.40625\ \mu\text{m}$ per pixel horizontally and vertically.
    
- **Axial resolution ($z = 1.625\ \mu\text{m}$):**
    
    The microscope takes focal slices at physical steps of $1.625\ \mu\text{m}$ apart along the depth axis.
## What would happen if you don't account this distortion ?
If you calculate Euclidean distance using raw voxel grid units instead of physical units, **distances along $z$ are severely underestimated**, which distorts the $7.0\ \mu\text{m}$ threshold evaluation

**Example:** Suppose a predicted node P and a ground-truth node G are separated by 5 voxels along z and 0 voxels along x and y.
*If you* calculated the Euclidean distance using the raw voxel unit $$\text{Physical distance} = \sqrt{0^2 + 0^2 + 5^2} =5 $$
meaning that predicted node would be considered as a true positive. But *if you account* for the physical distortion introduced by the measurement (cropping one axis with respect to the other) the new distance would be
$$\text{Physical distance} = \sqrt{0^2 + 0^2 + (5*1.625)^2} =8.125 $$

In this case, the distance is larger than the threshold ($7 \mu \text{m}$) and this node would be categorized as false positve.
