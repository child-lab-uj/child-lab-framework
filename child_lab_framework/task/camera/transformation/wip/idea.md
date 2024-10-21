## Calibration step

A YAML file (see input_example.yaml) containing:
- preset label (string name of the whole preset)
- list of filepaths to calibration videos with their respective camera labels
- list of filepaths to transformation videos with their respective camera labels

### 1. Input:
- preset label
- list[calibration video + camera label]
- list[transformation video + camera label]


### 2. Output:
Config file(s) with:
- Camera intrinsics
- Depth denormalization constant
- Transforms between cameras to the "root" camera

A YAML file containing:
- Camera intrinsics 

## Post proessing

### 1. 
### 2. :
Reconstruct the pointcloud from videos
 