using Aardvark.Base;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Aardvark.OpenCV
{
    internal class RobustEstimation<Data, Model> where Model : IValidity, new()
    {
        private Func<Data[], Model> m_createModel;
        private Func<Data, Model, double> m_computeResidual;
        private int m_numSamples;

        public RobustEstimation(int numSamples, Func<Data[], Model> createModel, Func<Data, Model, double> computeResidual)
        {
            m_createModel = createModel;
            m_computeResidual = computeResidual;
            m_numSamples = numSamples;
        }

        private int GetNumberOfIterations(double expectedOutlierRatio, double probability)
        {
            return (int)((1 - probability).Log() / (1 - (1 - expectedOutlierRatio).Pow(m_numSamples)).Log());
        }

        #region Ransac

        public Model SolveWithRansac(Data[] data, double maxError, double expectedOutlierRatio, double probability, out int[] inliers)
        {
            var maxIt = GetNumberOfIterations(expectedOutlierRatio, probability);
            return SolveWithRansac(data, maxError, maxIt, probability, out inliers);
        }

        public Model SolveWithRansac(Data[] data, double maxError, int iterations, double probability, out int[] inliers)
        {
            if (data.Length < m_numSamples)
                throw new InvalidOperationException("Ransac: provided less than the minimum number of required samples");

            var rnd = new Random();

            inliers = new int[0];
            var bestModel = new Model();

            for (var it = 0; it < iterations; it++)
            {
                var indices = new HashSet<int>();
                while (indices.Count < m_numSamples)
                {
                    var i = rnd.Next(data.Length);
                    if (!indices.Contains(i)) indices.Add(i);
                }
                var samples = indices.Select(i => data[i]).ToArray();

                var model = m_createModel(samples);
                if (model.IsValid)
                {
                    var errors = data.Map(d => m_computeResidual(d, model));
                    var newInliers = (0).UpToExclusive(data.Length).Where(i => errors[i] < maxError).ToArray();
                    if (newInliers.Length > inliers.Length)
                    {
                        inliers = newInliers;
                        bestModel = model;

                        //adapt number of iterations if more inliers found than stated by outlierRatio
                        var outlierRatio = 1.0 - (double)inliers.Length / data.Length;
                        var newIterations = GetNumberOfIterations(outlierRatio, probability);
                        if (newIterations < iterations)
                        {
                            iterations = newIterations;
                        }
                    }
                }
            }

            return bestModel;
        }

        #endregion

        #region Prosac

        public Model SolveWithProsac(Data[] data, double[] qualities, double maxError, double expectedOutlierRatio, double probability, out int[] inliers)
        {
            var maxIt = GetNumberOfIterations(expectedOutlierRatio, probability);
            return SolveWithProsac(data, qualities, maxError, maxIt, probability, out inliers);
        }

        public Model SolveWithProsac(Data[] data, double[] qualities, double maxError, int iterations, double probability, out int[] inliers)
        {
            if (data.Length < m_numSamples)
                throw new InvalidOperationException("Prosac: provided less than the minimum number of required samples");
            if (data.Length != qualities.Length)
                throw new InvalidOperationException("Prosac: wrong number of quality values");

            var U_N = (0).UpToExclusive(qualities.Length).OrderByDescending(i => qualities[i]).ToArray();

            var rnd = new Random();

            inliers = new int[0];
            Model bestModel = new Model();
            int t = 0;
            int m = m_numSamples;
            int n = m;
            int n_star = U_N.Length;
            int T_N = 200000;
            float T_n = T_N;
            for (int i = 0; i < m; ++i)
                T_n *= (float)(m - i) / (float)(U_N.Length - i);
            float T_prime_n = 1;
            int I_N_best = 0;

            for (int it = 0; it < iterations; it++)
            {
                //choice of hypothesis generation set
                t += 1;
                if (t == T_prime_n && n < n_star)
                {
                    n = n + 1;

                    float T_n_minus_1 = T_n;
                    T_n *= (float)(n + 1) / (float)(n + 1 - m);
                    T_prime_n += Fun.Ceiling(T_n - T_n_minus_1);
                }

                //semi-random sample of size m
                var indices = new List<int>();
                if (T_prime_n < t) indices.Add(n - 1);
                while (indices.Count < m_numSamples)
                {
                    var i = rnd.Next(n - 1);
                    if (!indices.Contains(i)) indices.Add(i);
                }

                //model parameter estimation
                var samples = indices.Select(i => data[U_N[i]]).ToArray();
                var model = m_createModel(samples);

                //model verification (against all data)
                var errors = U_N.Select(idx => m_computeResidual(data[idx], model)).ToArray();
                var newInliers = (0).UpToExclusive(errors.Length).Where(i => errors[i] < maxError).ToArray();
                int I_N = newInliers.Length;
                if (I_N > I_N_best)
                {
                    I_N_best = I_N;
                    inliers = newInliers;
                    bestModel = model;

                    //todo: n_star optimization to reduce number of samples
                    iterations = Fun.Min(iterations, GetNumberOfIterations(1 - (double)newInliers.Length / U_N.Length, probability));
                }
            }

            //convert to original indices
            inliers = inliers.Select(i => U_N[i]).ToArray();
            return bestModel;
        }

        #endregion
    }
}
