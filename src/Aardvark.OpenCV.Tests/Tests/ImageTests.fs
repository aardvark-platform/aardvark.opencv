namespace Aardvark.OpenCV.Tests

open Aardvark.Base
open NUnit.Framework
open FsUnit

module ``Image Processing Tests`` =

    let private rnd = RandomSystem()

    module private PixImage =

        let random (size : V2i) =
            let pi = PixImage<float32>(Col.Format.RGBA, size)
            for c in pi.ChannelArray do
                c.SetByIndex(ignore >> rnd.UniformFloat) |> ignore

            pi

    [<DatapointSource>]
    let filters = [|
        ImageInterpolation.Near
        ImageInterpolation.Linear
        ImageInterpolation.Cubic
        ImageInterpolation.Lanczos
    |]

    [<SetUp>]
    let init() =
        Aardvark.Init()

    [<Theory>]
    let ``Scaling`` (filter: ImageInterpolation) (sub: bool) =
        let size = V2i(512) + rnd.UniformV2i(1024)
        let scaleFactor = V2d(0.2345, 1.6789)

        let src =
            let pi = PixImage.random size
            if sub then
                let min = 10
                let offset = rnd.UniformV2i(size - min - 1)
                let size = min + rnd.UniformV2i(size - offset - min - 1)
                new PixImage<float32>(pi.Format, pi.Volume.SubVolume(offset.XYO, V3i(size, pi.ChannelCount)))
            else
                pi

        let result = Aardvark.OpenCV.PixProcessor.Instance.Scale(src, scaleFactor, filter)
        let reference = Aardvark.Base.PixProcessor.Instance.Scale(src, scaleFactor, filter)

        let psnr = PixImage.peakSignalToNoiseRatio result reference
        psnr |> should be (greaterThan 20.0)