namespace Aardvark.OpenCV.Benchmarks

open System.Reflection
open BenchmarkDotNet.Running;
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Jobs
open BenchmarkDotNet.Toolchains

module Program =

    [<EntryPoint>]
    let main argv =

        let cfg =
            let job = Job.ShortRun.WithToolchain(InProcess.Emit.InProcessEmitToolchain.Instance)
            ManualConfig.Create(DefaultConfig.Instance).WithOptions(ConfigOptions.DisableOptimizationsValidator).AddJob(job)

        BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(argv, cfg) |> ignore

        0