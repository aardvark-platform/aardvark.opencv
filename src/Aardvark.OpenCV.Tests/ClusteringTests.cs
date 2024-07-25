using Aardvark.Base;
using NUnit.Framework;

namespace Aardvark.OpenCV.Tests
{
    [TestFixture]
    internal class ClusteringTests
    {
        private static Vector<float> Vec(float v) => new([v, v, v]);

        [OneTimeSetUp]
        public void Init()
        {
            Aardvark.Base.Aardvark.Init();
        }

        [Test]
        public void KMeans()
        {
            Vector<float>[] data = { Vec(-1.2f), Vec(-1.3f), Vec(-1.4f), Vec(2.3f), Vec(2.4f) };
            OpenCVKMeansClustering.ClusterKMeans(data, 2, 1, false, out var clusters, out var centers);
            Assert.AreEqual(new int[] { 0, 0, 0, 1, 1 }, clusters);
        }
    }
}
