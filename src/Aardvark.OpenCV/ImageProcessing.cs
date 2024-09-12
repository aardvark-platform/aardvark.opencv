using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Aardvark.Base;
using OpenCvSharp;
using CvMat = OpenCvSharp.Mat;

namespace Aardvark.OpenCV
{
    #region Utilities

    internal readonly struct GCHandleDiposable : IDisposable
    {
        public GCHandle Handle { get; }

        public IntPtr AddrOfPinnedObject() => Handle.AddrOfPinnedObject();

        public GCHandleDiposable(GCHandle handle) { Handle = handle; }

        public void Dispose() => Handle.Free();
    }

    internal static class Extensions
    {
        public static GCHandleDiposable Pin(this object obj)
            => new (GCHandle.Alloc(obj, GCHandleType.Pinned));

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

        public static MatType ToMatType(this Type type, int channels)
        {
            if (matTypes.TryGetValue(type, out var toMatType)) return toMatType(channels);
            else throw new NotSupportedException($"Channel type {type} is not supported.");
        }

        public static InterpolationFlags ToInterpolationFlags(this ImageInterpolation interpolation, bool exact)
            => interpolation switch
            {
                ImageInterpolation.Near when exact   => (InterpolationFlags)6,
                ImageInterpolation.Near              => InterpolationFlags.Nearest,
                ImageInterpolation.Linear when exact => InterpolationFlags.LinearExact,
                ImageInterpolation.Linear            => InterpolationFlags.Linear,
                ImageInterpolation.Cubic             => InterpolationFlags.Cubic,
                ImageInterpolation.Lanczos           => InterpolationFlags.Lanczos4,
                _ => throw new NotSupportedException($"Filter {interpolation} is not supported.")
            };

        public static BorderTypes ToBorderTypes(this ImageBorderType borderType)
            => borderType switch
            {
                ImageBorderType.Const  => BorderTypes.Constant,
                ImageBorderType.Repl   => BorderTypes.Replicate,
                ImageBorderType.Wrap   => BorderTypes.Wrap,
                ImageBorderType.Mirror => BorderTypes.Reflect,
                _ => throw new NotSupportedException($"Border type {borderType} is not supported.")
            };

        private static GCHandleDiposable ToMat(this Array data, V2l size, V2l delta, long firstIndex, MatType matType, int elementSize, out CvMat result)
        {
            var gc = data.Pin();

            var ptr = IntPtr.Add(gc.AddrOfPinnedObject(), (int)firstIndex * elementSize);
            var sizei = size.ToV2i();
            var deltai = delta.YX * elementSize;

            result = CvMat.FromPixelData(sizei.YX.ToArray(), matType, ptr, deltai.ToArray());
            return gc;
        }

        public static GCHandleDiposable ToMat<T>(this Matrix<T> matrix, MatType matType, int elementSize, out CvMat result)
            => matrix.Data.ToMat(matrix.Size, matrix.Delta, matrix.FirstIndex, matType, elementSize, out result);

        public static GCHandleDiposable ToMat<T>(this Volume<T> volume, MatType matType, int elementSize, out CvMat result)
            => volume.Data.ToMat(volume.Size.XY, volume.Delta.XY, volume.FirstIndex, matType, elementSize, out result);
    }

    #endregion

    public sealed class PixProcessor : IPixProcessor
    {
        public string Name => "OpenCV";

        public PixProcessorCaps Capabilities => PixProcessorCaps.Scale | PixProcessorCaps.Remap;

        [OnAardvarkInit]
        public static void Init()
        {
            PixImage.AddProcessor(Instance);
        }

        #region Scale

        public PixImage<T> Scale<T>(PixImage<T> image, V2d scaleFactor, ImageInterpolation interpolation)
        {
            var src = image.Volume;

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

            var matType = typeof(T).ToMatType(image.ChannelCount);
            var elementSize = typeof(T).GetCLRSize();

            using var _src = src.ToMat(matType, elementSize, out var srcMat);
            using var _dst = dst.ToMat(matType, elementSize, out var dstMat);

            Cv2.Resize(
                srcMat, dstMat,
                new Size((int)dst.SX, (int)dst.SY),
                interpolation: interpolation.ToInterpolationFlags(true)
            );

            return new (image.Format, dst);
        }

        #endregion

        #region Rotate

        public PixImage<T> Rotate<T>(PixImage<T> image, double angleInRadians, bool resize, ImageInterpolation interpolation,
                                     ImageBorderType borderType = ImageBorderType.Const,
                                     T border = default)
            => null;

        #endregion

        #region Remap

        // TODO: Change the interface to accept T[] as border value
        /// <inheritdoc cref="Remap{T}(PixImage{T}, Matrix{float}, Matrix{float}, ImageInterpolation, ImageBorderType, T)"/>
        public PixImage<T> Remap<T>(PixImage<T> image, Matrix<float> mapX, Matrix<float> mapY, ImageInterpolation interpolation,
                                    ImageBorderType borderType = ImageBorderType.Const,
                                    T[] border = default)
        {
            var src = image.Volume;

            if (!src.HasImageWindowLayout())
            {
                throw new ArgumentException($"Volume must be in image layout (Delta = {src.Delta}).");
            }

            if (mapX.Size != mapY.Size)
            {
                throw new ArgumentException($"Size of coordinate maps must match (mapX: {mapX.Size}, mapY: {mapY.Size}).");
            }

            var matType = typeof(T).ToMatType(image.ChannelCount);
            var elementSize = typeof(T).GetCLRSize();

            var dstSize = new V3l(mapX.Size, image.ChannelCountL);
            var dst = dstSize.CreateImageVolume<T>();

            using var _s = src.ToMat(matType, elementSize, out var srcMat);
            using var _d = dst.ToMat(matType, elementSize, out var dstMat);
            using var _x = mapX.ToMat(MatType.CV_32FC1, 4, out var mapXMat);
            using var _y = mapY.ToMat(MatType.CV_32FC1, 4, out var mapYMat);

            Scalar borderValue = new();
            if (borderType == ImageBorderType.Const && border != null)
            {
                var order = image.Format.ChannelOrder();

                for (var i = 0; i < Fun.Min(border.Length, 4); i++)
                    borderValue[i] = Convert.ToDouble(border[order[i]]);
            }

            Cv2.Remap(
                srcMat, dstMat, mapXMat, mapYMat,
                interpolation.ToInterpolationFlags(false),
                borderType.ToBorderTypes(),
                borderValue
            );

            return new (image.Format, dst);
        }

        public PixImage<T> Remap<T>(PixImage<T> image, Matrix<float> mapX, Matrix<float> mapY, ImageInterpolation interpolation,
                                    ImageBorderType borderType,
                                    T border)
            => Remap(image, mapX, mapY, interpolation, borderType, new T[] { border, border, border, border });

        #endregion

        private PixProcessor() { }

        private static readonly Lazy<PixProcessor> _instance = new(() => new PixProcessor());

        public static PixProcessor Instance => _instance.Value;
    }
}
