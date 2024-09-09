namespace Aardvark.OpenCV.Tests

open Aardvark.Base
open Aardvark.OpenCV
open NUnit.Framework
open FsUnit

module ``Fitting Tests`` =

    [<SetUp>]
    let init() =
        Aardvark.Init()

    [<Test>]
    let ``Plane3d least squares``() =
        let data = [| V3d(1.0, 2.0, -1.0); V3d(5.0, 4.0, -1.0); V3d(23.0, 100.0, -1.0) |]
        let plane = data.FitPlane3dLeastSquares();
        plane |> should equal (Plane3d(V3d.ZAxis, -1))