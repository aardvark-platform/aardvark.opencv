namespace Aardvark.OpenCV.Benchmarks

open Aardvark.Base
open Aardvark.OpenCV
open BenchmarkDotNet.Attributes

module ``Image Processing Benchmarks`` =

    type Scaling() =

        [<DefaultValue; Params(128, 1024, 2048, 4096)>]
        val mutable Size : int

        [<DefaultValue; Params(ImageInterpolation.Linear, ImageInterpolation.Cubic, ImageInterpolation.Lanczos)>]
        val mutable Interpolation : ImageInterpolation

        let mutable image = null

        let scaleFactor = V2d(0.234, 0.894)

        [<GlobalSetup>]
        member x.Setup() =
            Aardvark.Init()
            let rnd = RandomSystem 0
            image <- new PixImage<float32>(Col.Format.RGBA, x.Size, x.Size)
            image.Volume.SetByIndex(ignore >> rnd.UniformFloat) |> ignore

        [<Benchmark(Description = "Aardvark (Tensors)", Baseline = true)>]
        member x.AardvarkTensors() =
            let volume = Aardvark.Base.TensorExtensions.Scaled(image.Volume, scaleFactor, x.Interpolation)
            PixImage<float32>(image.Format, volume)

        [<Benchmark>]
        member x.OpenCV() =
            ImageProcessing.ScaledOpenCV(image, scaleFactor, x.Interpolation)