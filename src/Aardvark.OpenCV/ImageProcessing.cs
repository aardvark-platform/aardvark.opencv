using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Aardvark.Base;
using OpenCvSharp;
using CvMat = OpenCvSharp.Mat;

namespace Aardvark.OpenCV
{
    internal readonly struct GCHandleDiposable : IDisposable
    {
        public GCHandle Handle { get; }

        public IntPtr AddrOfPinnedObject() => Handle.AddrOfPinnedObject();

        public GCHandleDiposable(GCHandle handle) { Handle = handle; }

        public void Dispose() => Handle.Free();
    }

    internal static class PinningExtensions
    {
        public static GCHandleDiposable Pin(this object obj)
            => new (GCHandle.Alloc(obj, GCHandleType.Pinned));
    }

    public static class ImageProcessing
    {
        [OnAardvarkInit]
        public static void Init()
        {
            PixImage<byte>.SetScaledFun(ScaledOpenCV);
            PixImage<sbyte>.SetScaledFun(ScaledOpenCV);
            PixImage<ushort>.SetScaledFun(ScaledOpenCV);
            PixImage<short>.SetScaledFun(ScaledOpenCV);
            PixImage<uint>.SetScaledFun(ScaledOpenCV);
            PixImage<int>.SetScaledFun(ScaledOpenCV);
            PixImage<ulong>.SetScaledFun(ScaledOpenCV);
            PixImage<long>.SetScaledFun(ScaledOpenCV);
            PixImage<Half>.SetScaledFun(ScaledOpenCV);
            PixImage<float>.SetScaledFun(ScaledOpenCV);
            PixImage<double>.SetScaledFun(ScaledOpenCV);
        }

        private static readonly Dictionary<Type, Func<int, MatType>> matTypes = new()
        {
            { typeof(byte),   MatType.CV_8UC },
            { typeof(sbyte),  MatType.CV_8SC },
            { typeof(short),  MatType.CV_16UC },
            { typeof(ushort), MatType.CV_16SC },
            { typeof(int),    MatType.CV_32SC },
            { typeof(float),  MatType.CV_32FC },
            { typeof(double), MatType.CV_64FC },
        };

        private static MatType ToMatType(this Type type, int channels)
        {
            if (matTypes.TryGetValue(type, out var toMatType)) return toMatType(channels);
            else throw new NotSupportedException($"Channel type {type} is not supported.");
        }

        private static readonly Dictionary<ImageInterpolation, InterpolationFlags> interpolationFlags = new()
        {
            { ImageInterpolation.Near,    (InterpolationFlags)6 }, // INTER_NEAREST_EXACT
            { ImageInterpolation.Linear,  InterpolationFlags.Linear },
            { ImageInterpolation.Cubic,   InterpolationFlags.Cubic },
            { ImageInterpolation.Lanczos, InterpolationFlags.Lanczos4 },
        };

        private static InterpolationFlags ToInterpolationFlags(this ImageInterpolation interpolation)
        {
            if (interpolationFlags.TryGetValue(interpolation, out InterpolationFlags flags)) return flags;
            else throw new NotSupportedException($"Filter {interpolation} is not supported.");
        }

        public static Volume<T> ScaledOpenCV<T>(this Volume<T> src, V2d scaleFactor, ImageInterpolation interpolation)
        {
            if (!src.HasImageWindowLayout())
            {
                throw new ArgumentException($"Volume must be in image layout (Delta = {src.Delta}).");
            }

            var newSize = new V3l((V2l)(V2d.Half + scaleFactor * (V2d)src.Size.XY), src.Size.Z);
            if (newSize.AnySmallerOrEqual(0))
            {
                throw new ArgumentException($"Scaled size must be positive (is {newSize}).");
            }

            var dst = newSize.CreateImageVolume<T>();

            var matType = typeof(T).ToMatType((int)src.SZ);
            var elementSize = typeof(T).GetCLRSize();

            using var srcGC = src.Data.Pin();
            var srcPtr = IntPtr.Add(srcGC.AddrOfPinnedObject(), (int)src.FirstIndex * elementSize);
            var srcSize = src.Size.XY.ToV2i();
            var srcDelta = src.Delta.YX * elementSize;

            using var dstGC = dst.Data.Pin();
            var dstPtr = dstGC.AddrOfPinnedObject();
            var dstSize = dst.Size.XY.ToV2i();

            var srcMat = CvMat.FromPixelData(srcSize.YX.ToArray(), matType, srcPtr, srcDelta.ToArray());
            var dstMat = CvMat.FromPixelData(dstSize.Y, dstSize.X, matType, dstPtr);
            Cv2.Resize(srcMat, dstMat, new Size(dstSize.X, dstSize.Y), interpolation: interpolation.ToInterpolationFlags());

            return dst;
        }

        public static PixImage<T> ScaledOpenCV<T>(this PixImage<T> src, V2d scaleFactor, ImageInterpolation interpolation)
            => new (src.Format, src.Volume.ScaledOpenCV(scaleFactor, interpolation));
    }
}
