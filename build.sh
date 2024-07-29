#!/bin/sh

dotnet tool restore
dotnet paket restore
dotnet build src/Aardvark.OpenCV.sln