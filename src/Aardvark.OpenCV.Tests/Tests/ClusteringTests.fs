namespace Aardvark.OpenCV.Tests

open Aardvark.Base
open Aardvark.OpenCV
open NUnit.Framework
open FsUnit

module ``Clustering Tests`` =

    let private vec value =
        Vector<float32> [| value; value; value |]

    [<SetUp>]
    let init() =
        Aardvark.Init()

    [<Test>]
    let ``K-means``() =
        let data = [| vec -1.2f; vec -1.3f; vec -1.4f; vec 2.3f; vec 2.4f |]

        let mutable clusters = Array.empty
        let mutable centers = Array.empty
        OpenCVKMeansClustering.ClusterKMeans(data, 2, 1, false, &clusters, &centers);

        clusters |> should equal [| 0; 0; 0; 1; 1 |]

