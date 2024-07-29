using Aardvark.Base;
using OpenCvSharp;
using CvMat = OpenCvSharp.Mat;

namespace Aardvark.OpenCV
{
    public static class OpenCVKMeansClustering
    {
        #region Clustering

        /// <summary>
        /// Standard K-means Clustering
        /// </summary>
        /// <param name="data">data array, with each point as vector</param>
        /// <param name="k">number of clusters</param>
        /// <param name="attempts">number of attempts</param>
        /// <param name="zeroMean">if true, data is shifted to zero mean</param>
        /// <param name="centers">returns the centroids of the clusters</param>
        /// <param name="clusters">index array with cluster index per data point</param>
        public static void ClusterKMeans(Vector<float>[] data, int k, int attempts, bool zeroMean, out int[] clusters, out Vector<float>[] centers)
        {
            int m = data.Length;      // number of points = ROWS = Y
            int n = (int)data[0].Count; // dim per point = COLS = X
            var A = new Matrix<float>(n, m);
            var B = new Matrix<int>(1, m);
            var C = new Matrix<float>(n, k);
            var mean = new Vector<float>(n);

            //-- accummualtion
            for (int j = 0; j < m; j++)
            {
                var v = data[j];
                for (int i = 0; i < n; i++)
                {
                    mean[i] += v[i];
                    A[i, j] = v[i];
                }
            }
            if (zeroMean)
            {
                for (int i = 0; i < n; i++)
                    mean[i] /= m;

                for (int j = 0; j < m; j++)
                    for (int i = 0; i < n; i++)
                        A[i, j] -= mean[i];
            }

            //-- termination criteria
            var term = new TermCriteria(CriteriaTypes.MaxIter | CriteriaTypes.Eps, 1000, 0.0001);

            var mA = CvMat.FromPixelData(m, n, MatType.CV_32FC1, A.Array);
            var mB = CvMat.FromPixelData(m, 1, MatType.CV_32SC1, B.Array);
            var mC = CvMat.FromPixelData(k, n, MatType.CV_32FC1, C.Array);

            //Report.BeginTimed("Computing Range K-Means ....");

            var cvCenters = OutputArray.Create(mC);
            double compactness = OpenCvSharp.Cv2.Kmeans(mA, k, mB, term, attempts, KMeansFlags.RandomCenters, cvCenters);
            //Report.End("Done.");

            //Report.Line("compactness: " + compactness);

            centers = new Vector<float>[k];
            for (int i = 0; i < k; i++)
            {
                centers[i] = C.SubXVectorWindow(i).CopyWindow();
            }
            //centers = B.Data.Select(x => new Vector<float>(C.SubXVectorZero(x).Copy().Data, data[0].Size)).ToArray();
            //var cvolume = new Volume<float>(cdata, data[0].Size);
            clusters = B.Data;//.AsVectorZero();
        }

        #endregion
    }
}