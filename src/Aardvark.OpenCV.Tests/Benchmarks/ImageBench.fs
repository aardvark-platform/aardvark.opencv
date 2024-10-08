﻿namespace Aardvark.OpenCV.Benchmarks

open Aardvark.Base
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
            Aardvark.Base.PixProcessor.Instance.Scale(image, scaleFactor, x.Interpolation)

        [<Benchmark>]
        member x.OpenCV() =
            Aardvark.OpenCV.PixProcessor.Instance.Scale(image, scaleFactor, x.Interpolation)