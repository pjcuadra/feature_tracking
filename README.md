# Feature Tracking

This a Feature Tracking Computer Vision project using openCV. It's intended
to be applied to the *Mapping a new road with senseFly Corridor* data set
available at [senseFly](https://www.sensefly.com/drones/example-datasets.html).

# Dependencies

In order to build the project first install OpenCV 3.3 with support for the
non-free modules from the *OpenCV contrib repository* as specified [here](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html).

# Input images

You can download the *Mapping a new road with senseFly Corridor* data set from [here](https://senseflycom.s3.amazonaws.com/datasets/rc177-corridor/rgb-jpg.zip).
The data set comes in a `zip` file. Therefore, to use the images first unzip
the archive.

# Build the code

To build the code you'll simply have to run;

```
  cd <project-root-dir>
  mkdir build
  cd build
  cmake ..
  make
```

# Running tests

This project uses *Google Test* for as testing framework. After compiling
you can run our tests by running;

```
  cd <project-root-dir>
  cd build
  ./tests_exec
```

# Running Feature Extraction Demo

TBD

# Running Feature Tracking Demo

After compiling you can run the tracking demo by running;

```
  cd <project-root-dir>
  cd build
  ./tracking_demo -indir=<path-to-input> [-outdir=<path-to-output> -show -extract -match -v -finder=<org|surf> -help]
```

## Options

* *help* - displays help
* *v* - enable verbose mode.
* *indir* - path to the input data set
* *outdir* - path to where the output will be written
* *show* - enables the GUI and displays images in a pair basis
* *extract* - enable feature extraction. If not specified the features will be
  read from `<project-root-dir>/build/features.yml`. If the `features.yml` file
  doesn't exist then the extraction is automatically enabled.
* *match* - enable feature matching. If not specified the matches will be
  read from `<project-root-dir>/build/features.yml`. If the `features.yml` file
  doesn't exist then the matching is automatically enabled.
* *finder* - specified the feature finder to be used. Only SURF was tested.

## Output GUI

The GUI is enabled by adding the *show* parameter. The GUI will open 4 different
windows described below;

1. *Image 1:* First image with the extracted feature points drawn as rich
   points.
2. *Image 2:* Second image with the extracted feature points drawn as rich
   points.
3. *Matches:* Both images with the extracted feature points drawn as rich
   points and the matches joining the points from one image to the other.
4. *Warped:* First image with the second image with the homography applied
   superposed.

## Output Directory

If the output directory is specified using *outdir* parameter then a separate
directory per image pair will be created named as the first image, but without
the extension. For example the image pair `EP-00-00012_0366_0001.JPG` and
`EP-00-00012_0366_0002.JPG` will create a directory named `EP-00-00012_0366_0001`
for storing their outputs. The directory will contain;

* `homography.yml` - with the obtained homography stored using OpenCV's
  `FileStorage`.
* `decomposedHomography.yml` - with the obtained homography, camera matrix,
  translational and rotational matrix. Also stored using OpenCV's `FileStorage`.
* `<first-image-file-name>` - with the first image file scaled as used in the
  algorithm.
* `<second-image-file-name>` - with the second image file scaled as used in the
  algorithm.
* `features_<first-image-file-name>` - with the first image and the found SIFT
  features drawn as rich points.
* `features_<second-image-file-name>` - with the second image and the found SIFT
  features drawn as rich points.
* `matches_<first-image-file-name>` - with the both image side-by-side, the
  found SIFT features drawn as rich points, and the drawn matches as lines.
* `warped_<first-image-file-name>` - with the superposition of the first image
  and the second with the homography applied.
