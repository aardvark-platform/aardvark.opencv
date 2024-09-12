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

        public static Scalar ToScalar<T>(this T[] color, Col.Format format)
        {
            var result = new Scalar();

            if (color != null)
            {
                var order = format.ChannelOrder();
                for (var i = 0; i < Fun.Min(color.Length, format.ChannelCount()); i++)
                    result[i] = Convert.ToDouble(color[order[i]]);
            }

            return result;
        }
    }

    #endregion

    public sealed class PixProcessor : IPixProcessor
    {
        public string Name => "OpenCV";

        public PixProcessorCaps Capabilities => PixProcessorCaps.Scale | PixProcessorCaps.Rotate | PixProcessorCaps.Remap;

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

        // TODO: Change the interface to accept T[] as border value
        /// <inheritdoc cref="Rotate{T}(PixImage{T}, double, bool, ImageInterpolation, ImageBorderType, T)"/>
        public PixImage<T> Rotate<T>(PixImage<T> image, double angleInRadians, bool resize, ImageInterpolation interpolation,
                                     ImageBorderType borderType = ImageBorderType.Const,
                                     T[] border = default)
        {
            var src = image.Volume;

            if (!src.HasImageWindowLayout())
            {
                throw new ArgumentException($"Volume must be in image layout (Delta = {src.Delta}).");
            }

            var matType = typeof(T).ToMatType(image.ChannelCount);
            var elementSize = typeof(T).GetCLRSize();

            var srcCenter = image.Size.ToV2d() * 0.5;

            // This already takes the handedness of the image coordinate system into account.
            // See: https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
            var rotMat =
                Cv2.GetRotationMatrix2D(
                    new Point2f((float)srcCenter.X, (float)srcCenter.Y),
                    -angleInRadians.DegreesFromRadians(),
                    1.0
                );

            var dstSize = image.Volume.Size;
            if (resize)
            {
                // Compute bounds of rotated image
                // See: https://stackoverflow.com/questions/3231176/how-to-get-size-of-a-rotated-rectangle
                var cos = rotMat.At<double>(0, 0);
                var sin = rotMat.At<double>(0, 1);
                var cosAbs = cos.Abs();
                var sinAbs = sin.Abs();
                dstSize.X = (long)(image.Width * cosAbs + image.Height * sinAbs + 0.5);
                dstSize.Y = (long)(image.Width * sinAbs + image.Height * cosAbs + 0.5);

                // Adjust transformation matrix for new center
                // We describe the inverse transformation (i.e. from dst to src).
                // Shift by -dstCenter -> rotate CW -> shift by srcCenter.
                // See: https://math.stackexchange.com/questions/2093314/rotation-matrix-of-rotation-around-a-point-other-than-the-origin
                var dstCenter = dstSize.XY.ToV2d() * 0.5;
                rotMat.At<double>(0, 2) = -dstCenter.X * cos - dstCenter.Y * sin + srcCenter.X;
                rotMat.At<double>(1, 2) =  dstCenter.X * sin - dstCenter.Y * cos + srcCenter.Y;
            }

            var dst = dstSize.CreateImageVolume<T>();

            using var _src = src.ToMat(matType, elementSize, out var srcMat);
            using var _dst = dst.ToMat(matType, elementSize, out var dstMat);

            Cv2.WarpAffine(
                srcMat, dstMat, rotMat,
                new Size((int)dst.SX, (int)dst.SY),
                interpolation.ToInterpolationFlags(false) | InterpolationFlags.WarpInverseMap,
                borderType.ToBorderTypes(),
                (borderType == ImageBorderType.Const) ? border.ToScalar(image.Format) : new()
            );

            return new(image.Format, dst);
        }

        public PixImage<T> Rotate<T>(PixImage<T> image, double angleInRadians, bool resize, ImageInterpolation interpolation,
                                     ImageBorderType borderType,
                                     T border)
            => Rotate(image, angleInRadians, resize, interpolation, borderType, new T[] { border, border, border, border });

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

            Cv2.Remap(
                srcMat, dstMat, mapXMat, mapYMat,
                interpolation.ToInterpolationFlags(false),
                borderType.ToBorderTypes(),
                (borderType == ImageBorderType.Const) ? border.ToScalar(image.Format) : new()
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
