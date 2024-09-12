namespace Aardvark.OpenCV.Tests

open System
open Aardvark.Base
open NUnit.Framework
open FsUnit

module ``Image Processing Tests`` =

    let private rnd = RandomSystem()

    module private PixImage =

        let private randomValues : Type -> (unit -> obj) =
            LookupTable.lookup [
                typeof<float32>, rnd.UniformFloat >> box
                typeof<uint8>,   rnd.UniformUInt >> uint8 >> box
                typeof<uint16>,  rnd.UniformUInt >> uint16 >> box
                typeof<int32>,   rnd.UniformInt >> box
            ]

        let private getRandomValue<'T>() =
            let get = typeof<'T> |> randomValues
            get() |> unbox<'T>

        let random<'T> (format: Col.Format) (size : V2i) =
            let pi = PixImage<'T>(format, size)
            for c in pi.ChannelArray do
                c.SetByIndex(ignore >> getRandomValue<'T>) |> ignore

            pi

        let generate<'T> (format: Col.Format) (sub: bool) =
            let size = V2i(512) + rnd.UniformV2i(1024)
            let pi = random<'T> format size
            if sub then
                let min = 10
                let offset = rnd.UniformV2i(size - min - 1)
                let size = min + rnd.UniformV2i(size - offset - min - 1)
                new PixImage<'T>(pi.Format, pi.Volume.SubVolume(offset.XYO, V3i(size, pi.ChannelCount)))
            else
                pi

    [<DatapointSource>]
    let filters = [|
        ImageInterpolation.Near
        ImageInterpolation.Linear
        ImageInterpolation.Cubic
        ImageInterpolation.Lanczos
    |]

    [<DatapointSource>]
    let borderTypes = [|
        ImageBorderType.Const
        ImageBorderType.Repl
        ImageBorderType.Wrap
        ImageBorderType.Mirror
    |]

    [<SetUp>]
    let init() =
        Aardvark.Init()

    [<Theory>]
    let ``Scale`` (filter: ImageInterpolation) (sub: bool) =
        let src = PixImage.generate<float32> Col.Format.RGBA sub
        let scaleFactor = V2d(0.2345, 1.6789)

        let result = Aardvark.OpenCV.PixProcessor.Instance.Scale(src, scaleFactor, filter)
        let reference = Aardvark.Base.PixProcessor.Instance.Scale(src, scaleFactor, filter)

        let psnr = PixImage.peakSignalToNoiseRatio result reference
        psnr |> should be (greaterThan 20.0)

    [<Theory>]
    let ``Remap`` (sub: bool) =
        let src = PixImage.generate<float32> Col.Format.RGBA sub

        let result =
            let mapX = Matrix<float32>(src.Size)
            mapX.SetByCoord(fun x _ -> float32 (x * 2L)) |> ignore

            let mapY = Matrix<float32>(src.Size)
            mapY.SetByCoord(fun _ y -> float32 (src.SizeL.Y - y - 1L)) |> ignore

            Aardvark.OpenCV.PixProcessor.Instance.Remap(src, mapX, mapY, ImageInterpolation.Near)

        let reference =
            let pi = src.Transformed(ImageTrafo.MirrorY)

            pi.Volume.SetByCoord(fun x y c ->
                if x * 2L < pi.SizeL.X then
                    pi.Volume.[x * 2L, y, c]
                else
                    0.0f
            ) |> ignore

            pi

        let psnr = PixImage.peakSignalToNoiseRatio result reference
        psnr |> should be (greaterThan 20.0)

    [<Theory>]
    let ``Remap (border type)`` (borderType: ImageBorderType) (sub: bool) =
        let src = PixImage.generate<uint16> Col.Format.BGRA sub
        let border = [| 1us; 2us; 3us; 4us |]

        let result =
            let mapX = Matrix<float32>(src.Size)
            let mapY = Matrix<float32>(src.Size)

            if borderType = ImageBorderType.Const then
                mapX.Set -1.0f |> ignore
            else
                mapX.SetByCoord(fun x _ -> float32 (src.SizeL.X + x)) |> ignore
                mapY.SetByCoord(fun _ y -> float32 y) |> ignore

            Aardvark.OpenCV.PixProcessor.Instance.Remap(src, mapX, mapY, ImageInterpolation.Near, borderType, border)

        let expected =
            match borderType with
            | ImageBorderType.Const ->
                let pi = new PixImage<uint16>(result.Format, result.Size)
                pi.GetMatrix<C4us>().Set(C4us border) |> ignore
                pi

            | ImageBorderType.Repl ->
                let pi = new PixImage<uint16>(result.Format, result.Size)
                pi.Volume.SetByCoord(fun _ y c -> src.Volume.[src.Size.X - 1, y, c]) |> ignore
                pi

            | ImageBorderType.Wrap ->
                src

            | ImageBorderType.Mirror ->
                src.Transformed(ImageTrafo.MirrorX)

            | _ -> raise <| NotImplementedException()

        let psnr = PixImage.peakSignalToNoiseRatio result expected
        psnr |> should equal infinity

    [<Theory>]
    let ``Rotate`` (resize: bool) (sub: bool) =
        let src = PixImage.generate<uint8> Col.Format.BGRA sub
        let angle = -Constant.PiTimesFour + rnd.UniformDouble() * Constant.PiTimesFour * 2.0

        let rotated =
            Aardvark.OpenCV.PixProcessor.Instance.Rotate(src, angle, resize, ImageInterpolation.Linear, border = [| 255uy; 0uy; 0uy; 255uy |])

        let remapped =
            let dstSize =
                if resize then
                    let srcSize = V2d src.Size.XY
                    let cos = abs <| cos angle
                    let sin = abs <| sin angle

                    V2l(
                        int64 (srcSize.X * cos + srcSize.Y * sin + 0.5),
                        int64 (srcSize.X * sin + srcSize.Y * cos + 0.5)
                    )

                else
                    src.SizeL

            let mapX = Matrix<float32>(dstSize)
            let mapY = Matrix<float32>(dstSize)

            // Note: Positive angle -> clockwise rotation in the image coordinate system
            let mat =
                let srcCenter = v2f src.Size * 0.5f
                let dstCenter = v2f dstSize * 0.5f
                M33f.Translation srcCenter * M33f.RotationZ(float32 angle) * M33f.Translation -dstCenter

            mapX.SetByCoord(fun x y -> V2l(x, y) |> v2f |> Mat.transformPos mat |> Vec.x) |> ignore
            mapY.SetByCoord(fun x y -> V2l(x, y) |> v2f |> Mat.transformPos mat |> Vec.y) |> ignore

            Aardvark.OpenCV.PixProcessor.Instance.Remap(src, mapX, mapY, ImageInterpolation.Linear, border = [| 255uy; 0uy; 0uy; 255uy |])

        let psnr = PixImage.peakSignalToNoiseRatio rotated remapped
        psnr |> should be (greaterThan 20.0)