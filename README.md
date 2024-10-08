# Aardvark.OpenCV

![Publish](https://github.com/aardvark-platform/aardvark.opencv/workflows/Publish/badge.svg)
![Windows](https://github.com/aardvark-platform/aardvark.opencv/workflows/Windows/badge.svg)
![MacOS](https://github.com/aardvark-platform/aardvark.opencv/workflows/MacOS/badge.svg)
![Linux](https://github.com/aardvark-platform/aardvark.opencv/workflows/Linux/badge.svg)


[![NuGet](https://badgen.net/nuget/v/Aardvark.OpenCV)](https://www.nuget.org/packages/Aardvark.OpenCV/)
[![NuGet](https://badgen.net/nuget/dt/Aardvark.OpenCV)](https://www.nuget.org/packages/Aardvark.OpenCV/)

Contains algorithms and utilities using OpenCV via the [OpenCVSharp](https://github.com/shimat/opencvsharp) wrapper for the Aardvark Platform.

## Building
We use a minimal version of OpenCV (and OpenCVSharp) only supporting the `core` and `imgproc` modules. To build this version on Windows run `build_minimal_opencvsharp.ps1`. Other platforms are not supported at the moment.