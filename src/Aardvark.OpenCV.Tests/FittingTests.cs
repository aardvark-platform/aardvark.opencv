using Aardvark.Base;
using NUnit.Framework;

namespace Aardvark.OpenCV.Tests
{
    [TestFixture]
    internal class FittingTests
    {
        private static V3d Vec(float x, float y, float z) => new(x, y, z);

        [OneTimeSetUp]
        public void Init()
        {
            Aardvark.Base.Aardvark.Init();
        }

        [Test]
        public void FitPlane3dLeastSquares()
        {
            V3d[] data = { Vec(1, 2, -1), Vec(5, 4, -1), Vec(23, 100, -1) };
            var plane = data.FitPlane3dLeastSquares();
            Assert.AreEqual(new Plane3d(V3d.ZAxis, -1), plane);
        }
    }
}
