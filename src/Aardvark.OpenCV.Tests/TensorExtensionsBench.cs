using Aardvark.Base;
using BenchmarkDotNet.Attributes;

namespace Aardvark.OpenCV.Tests
{
    public class Scaling
    {
        [Params(128, 1024, 2048, 4096)]
        public int Size;

        [Params(ImageInterpolation.Linear, ImageInterpolation.Cubic)]
        public ImageInterpolation Interpolation;

        private PixImage<float> _image;

        private readonly V2d _scaleFactor = new (0.234, 0.894);

        [GlobalSetup]
        public void Init()
        {
            Aardvark.Base.Aardvark.Init();
            var rnd = new RandomSystem(0);
            _image = new PixImage<float>(Col.Format.RGBA, new V2i(Size, Size));
            _image.Volume.SetByIndex((_) => rnd.UniformFloat());
        }

        [Benchmark(Description = "Aardvark (Tensors)", Baseline = true)]
        public PixImage<float> AardvarkTensors()
        {
            var volume = Aardvark.Base.TensorExtensions.Scaled(_image.Volume, _scaleFactor, Interpolation);
            return new PixImage<float>(_image.Format, volume);
        }

        [Benchmark]
        public PixImage<float> OpenCV()
            => TensorExtensions.ScaledOpenCV(_image, _scaleFactor, Interpolation);
    }
}
