
$OriginalLocation = Get-Location

try {
    if (!(Test-Path -Path "build_minimal")) {  
        mkdir "build_minimal" > $null
    }
    Set-Location "build_minimal"

    # Clone OpenCV
    if (!(Test-Path -Path "opencv")) {
        git clone --branch 4.10.0 "https://github.com/opencv/opencv.git" "opencv"
    }

    Set-Location "opencv"

    # Build OpenCV
    if (Test-Path -Path "build") {
        Remove-Item -Recurse -Force "build"
    }
    mkdir "build" > $null
    Set-Location "build"

    cmake `
    -D CMAKE_BUILD_TYPE=RELEASE `
    -D BUILD_SHARED_LIBS=OFF `
    -D ENABLE_CXX11=ON `
    -D BUILD_EXAMPLES=OFF `
    -D BUILD_DOCS=OFF `
    -D BUILD_PERF_TESTS=OFF `
    -D BUILD_TESTS=OFF `
    -D BUILD_JAVA=OFF `
    -D BUILD_LIST=core,imgproc `
    -D BUILD_TIFF=OFF `
    -D BUILD_OPENJPEG=OFF `
    -D BUILD_JASPER=OFF `
    -D BUILD_JPEG=OFF `
    -D BUILD_PNG=OFF `
    -D BUILD_OPENEXR=OFF `
    -D BUILD_WEBP=OFF `
    -D BUILD_ITT=OFF `
    -D WITH_GSTREAMER=OFF `
    -D WITH_ADE=OFF `
    -D WITH_FFMPEG=OFF `
    -D WITH_V4L=OFF `
    -D WITH_1394=OFF `
    -D WITH_GTK=OFF `
    -D WITH_OPENEXR=OFF `
    -D WITH_PROTOBUF=OFF `
    -D WITH_QUIRC=OFF `
    -D WITH_VTK=OFF `
    -D WITH_WIN32UI=OFF `
    -D WITH_NVCUVID=OFF `
    -D WITH_NVCUVENC=OFF `
    -D WITH_JASPER=OFF `
    -D WITH_OPENJPEG=OFF `
    -D WITH_JPEG=OFF `
    -D WITH_TIFF=OFF `
    -D WITH_PNG=OFF `
    -D WITH_WEBP=OFF `
    -D WITH_V4L=OFF `
    -D WITH_DSHOW=OFF `
    -D WITH_OPENCLAMDFFT=OFF `
    -D WITH_OPENCLAMDBLAS=OFF `
    -D WITH_DIRECTX=OFF `
    -D WITH_MSMF=OFF `
    -D WITH_DIRECTML=OFF `
    -D WITH_IMGCODEC_HDR=OFF `
    -D WITH_IMGCODEC_SUNRASTER=OFF `
    -D WITH_IMGCODEC_PXM=OFF `
    -D WITH_IMGCODEC_PFM=OFF `
    -D WITH_OBSENSOR=OFF `
    -D WITH_ITT=OFF `
    -D OPENCV_ENABLE_NONFREE=OFF `
    -D CV_TRACE=OFF `
    ..

    cmake --build . --target INSTALL --config Release

    # # Clone and patch OpenCVSharp
    Set-Location "..\.."

    if (!(Test-Path -Path "opencvsharp")) {
        git clone "https://github.com/shimat/opencvsharp.git" "opencvsharp"
        Set-Location "opencvsharp"
        git reset --hard ca1f60877aff090de5e3e456c06c7827f33f364d
        git apply --reject --whitespace=fix "..\..\build_minimal_opencvsharp.patch"
    } else { 
        Set-Location "opencvsharp"
    }

    # Build OpenCVSharp
    if (Test-Path -Path "build") {
        Remove-Item -Recurse -Force "build"
    }
    mkdir "build" > $null
    Set-Location "build"

    $OpenCVPath = (Resolve-Path "..\..\opencv\build\install").Path
    cmake -D CMAKE_INSTALL_PREFIX=x64\Release -D OpenCV_STATIC=ON -D OpenCV_DIR=$OpenCVPath ..\src\
    cmake --build . --target INSTALL --config Release

    Copy-Item "x64\Release\bin\OpenCvSharpExtern.dll" -Destination "..\..\..\libs\Native\Aardvark.OpenCV\windows\AMD64"
    
} finally {
    Set-Location $OriginalLocation.Path
}