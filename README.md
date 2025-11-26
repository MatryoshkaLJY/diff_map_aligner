# Diff Map Aligner

A Python tool for batch image alignment and local difference detection using SIFT features and perspective transformation.

## Overview

This tool performs the following operations:

1. **Image Registration**: Aligns multiple overlay images to a reference image using SIFT feature matching and homography estimation
2. **Perspective Transformation**: Applies perspective transforms to warp overlay images to match the reference coordinate system
3. **Difference Detection**: Identifies local differences between reference and aligned images using SIFT descriptor comparison
4. **Multi-layer TIFF Generation**: Creates a multi-page TIFF file with reference image, difference highlights, and aligned overlays

## Features

- **SIFT-based Feature Matching**: Uses OpenCV's SIFT detector with FLANN matcher for robust feature correspondence
- **RANSAC Homography**: Robust homography estimation with outlier rejection
- **Grid Sampling**: Efficient difference detection via regular grid sampling
- **Feathered Difference Masks**: Gaussian blur-based edge feathering for smooth transitions
- **Multi-layer Output**: Saves results as multi-page TIFF with transparency support
- **Visual Preview**: Generates comparison plots with legends showing detected regions and differences

## Installation

### Setup

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python diff_map_aligner.py <path_to_reference_image> <path_to_overlays_folder>
```

### Arguments

- `<path_to_reference_image>`: Path to the reference/base image (e.g., `base.jpg`)
- `<path_to_overlays_folder>`: Path to folder containing overlay images to align (e.g., `overlays/`)

### Example

```bash
python diff_map_aligner.py base.jpg overlays/
```

This will:
1. Read `base.jpg` as the reference image
2. Process all supported images in the `overlays/` folder
3. Generate `output.tif` with aligned layers
4. Display an interactive comparison plot

## Configuration

Edit the parameters in the main execution block:

```python
batch_paste_with_preview_and_correct_tiff(
    sys.argv[1],
    sys.argv[2],
    "output.tif",
    sample_step=16,      # Sampling interval in pixels (larger = faster but coarser)
    diff_thresh=128      # SIFT descriptor distance threshold (larger = stricter)
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_step` | 16 | Pixel interval for grid sampling. Larger values are faster but less dense. |
| `diff_thresh` | 128 | SIFT descriptor L2 distance threshold for detecting differences. Higher values are stricter. |
| `output_tif` | `output.tif` | Output multi-page TIFF filename |

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- TIFF (`.tif`, `.tiff`)
- BMP (`.bmp`)

## Output

### Multi-page TIFF (`output.tif`)

- **Page 1**: Reference image (RGB)
- **Page 2+**: For each successfully matched overlay:
  - Difference highlight layer (RGBA with feathered alpha mask)
  - Aligned overlay layer (RGBA with content mask)

### Visualization Plot

A matplotlib figure showing:
- **Left panel**: Reference image
- **Right panel**: Composite preview with:
  - Colored borders indicating each overlay region
  - Red dots marking detected local differences
- **Legend**: Region identification and difference point count

## Algorithm Details

### Feature Matching

1. Detect SIFT keypoints in reference and overlay images
2. Use FLANN (Fast Library for Approximate Nearest Neighbors) to find descriptor matches
3. Apply Lowe's ratio test (0.82 threshold) to filter ambiguous matches

### Homography Estimation

1. Use RANSAC to estimate perspective transformation matrix
2. Require minimum 10 inliers for valid homography
3. Warp overlay image to reference coordinate system

### Difference Detection

1. Sample SIFT descriptors on a regular grid within the warped image
2. Compare descriptors with reference image at same locations
3. Flag points where L2 distance exceeds `diff_thresh`
4. Create difference mask by drawing circles at flagged points
5. Apply Gaussian blur for edge feathering

### Connected Components (Commented)

The code includes a union-find based connected component filter that can be enabled to remove isolated difference points and keep only the largest connected component.

## Performance Notes

- **Memory**: Loads full reference image into memory; processes overlays sequentially
- **Speed**: SIFT detection and matching typically dominate runtime
  - Adjust `sample_step` to control difference detection resolution
  - Larger `sample_step` = fewer sampled points = faster processing

## Troubleshooting

### "Reference image read failed"
- Verify file path is correct
- Ensure image format is supported
- Check file is not corrupted

### "Too few matches" or "RANSAC failed"
- Overlay image may not have enough overlapping features with reference
- Verify images show similar content
- Try adjusting SIFT parameters in code

### No difference points detected
- Increase `diff_thresh` if detection is too strict
- Verify images actually have differences
- Check `sample_step` isn't so large that sampling misses details

## Dependencies

- **opencv-python**: Computer vision algorithms (SIFT, homography, image processing)
- **numpy**: Numerical computations and array operations
- **matplotlib**: Result visualization
- **Pillow**: Multi-page TIFF I/O and image format conversion