# Test Images

This folder contains sample images from each class in the dataset for testing the WeedSense application.

## Sample Images

### Crop (Soybean) - Label: 1
- `soybean_sample_1.tif` - Healthy soybean crop
- `soybean_sample_2.tif` - Soybean field view

### Weed (Grass) - Label: 0
- `grass_weed_1.tif` - Grass weed in field
- `grass_weed_2.tif` - Grass vegetation

### Weed (Broadleaf) - Label: 0
- `broadleaf_weed_1.tif` - Broadleaf weed sample
- `broadleaf_weed_2.tif` - Broadleaf vegetation

### Other (Soil) - Label: 0
- `soil_sample_1.tif` - Bare soil
- `soil_sample_2.tif` - Soil surface

## Usage

1. Open the WeedSense web application
2. Navigate to the Detection page
3. Upload any of these test images
4. Click "Run Detection" to see the classification results

## Expected Results

- **Soybean samples** → Should classify as "Soybean (Crop)" with high confidence
- **Grass/Broadleaf/Soil samples** → Should classify as "Weed / Other" with high confidence

## Note

These are `.tif` format images. The web application supports TIF files and will display the filename instead of a preview (since browsers don't natively render TIF images).
