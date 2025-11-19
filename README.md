# Aardvark.OpenCV

[![Build](https://github.com/aardvark-platform/aardvark.opencv/actions/workflows/build.yml/badge.svg)](https://github.com/aardvark-platform/aardvark.opencv/actions/workflows/build.yml)
[![Publish](https://github.com/aardvark-platform/aardvark.opencv/actions/workflows/publish.yml/badge.svg)](https://github.com/aardvark-platform/aardvark.opencv/actions/workflows/publish.yml)
[![Nuget](https://img.shields.io/nuget/vpre/aardvark.opencv)](https://www.nuget.org/packages/aardvark.opencv/)
[![Downloads](https://img.shields.io/nuget/dt/aardvark.opencv)](https://www.nuget.org/packages/aardvark.opencv/)

Contains algorithms and utilities using OpenCV via the [OpenCVSharp](https://github.com/shimat/opencvsharp) wrapper for the Aardvark Platform.

## Building
We use a minimal version of OpenCV (and OpenCVSharp) only supporting the `core` and `imgproc` modules. To build this version on Windows run `build_minimal_opencvsharp.ps1`. Other platforms are not supported at the moment.
