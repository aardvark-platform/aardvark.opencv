using Aardvark.Base;
using OpenCvSharp;
using CvMat = OpenCvSharp.Mat;
using System;

namespace Aardvark.OpenCV
{
    public static class MatrixOpenCvExtensions
    {
        #region Vector-Matrix Operations

        /// <summary>
        /// Solves a linear system or least-squares problem.
        /// Parameters:
        ///
        ///     * A – The source matrix
        ///     * B – The right-hand part of the linear system
        ///     * X – The output solution
        ///     * method –
        ///
        ///       The solution (matrix inversion) method
        /// </summary>
        public static bool SolveLinear(this Matrix<double> A, Vector<double> b, out Vector<double> x, DecompTypes method)
        {
            //  Ax = b
            //   x = inv(A) * b
            // M*K x K*N = M*N
            int M = (int)A.SY; //Rows
            int K = (int)A.SX; //Cols
            int N = 1;

            x = new Vector<double>(K * N);
            var mA = CvMat.FromPixelData(M, K, MatType.CV_64FC1, A.Array);    //: A (M*K)
            var mX = CvMat.FromPixelData(K, N, MatType.CV_64FC1, x.Array);    //: x (K*N)
            var mB = CvMat.FromPixelData(M, N, MatType.CV_64FC1, b.Array);    //: b (M*N)

            return Cv2.Solve(mA, mB, mX, method);
        }

        /// <summary>
        /// Note: the U and V matrice are TRANSPOSED!!! Transpose back by your self if you need!
        ///
        ///A
        ///    Source M×N matrix.
        ///S
        ///    Resulting singular value matrix (M×N or N×N) or vector (N×1).
        ///U
        ///    Optional left orthogonal matrix (M×M or M×N). If CV_SVD_U_T is specified,
        ///    the number of rows and columns in the sentence above should be swapped.
        ///V
        ///    Optional right orthogonal matrix (N×N)
        ///flags
        ///    Operation flags; can be 0 or combination of the following:
        ///        * CV_SVD_MODIFY_A enables modification of matrix A during the operation. It speeds up the processing.
        ///        * CV_SVD_U_T means that the tranposed matrix U is returned. Specifying the flag speeds up the processing.
        ///        * CV_SVD_V_T means that the tranposed matrix V is returned. Specifying the flag speeds up the processing.
        ///
        /// It is the solution for homogeneous systems Ax=0,
        /// where x is the last column of the V matrix.
        ///
        /// M = U * S * V'
        /// M : [M x N] -> source, usually rectangular with M smaller N
        /// U : [M x M] -> Left Singular Vectors (in the Columns of U)
        /// V : [N x N] -> Right Singular Vectors (in the Columns of V)
        /// S : [M x N] -> Singular Values
        /// </summary>
        public static void ComputeSingularValues(this Matrix<double> A,
            out Matrix<double> VT, out Matrix<double> S)
        {
            ComputeSingularValuesVTS(A, out VT, out S);
        }

        /// <summary>
        /// Note: the U and V matrice are TRANSPOSED!!! Transpose back by your self if you need!
        ///
        ///A
        ///    Source M×N matrix.
        ///S
        ///    Resulting singular value matrix (M×N or N×N) or vector (N×1).
        ///U
        ///    Optional left orthogonal matrix (M×M or M×N). If CV_SVD_U_T is specified,
        ///    the number of rows and columns in the sentence above should be swapped.
        ///V
        ///    Optional right orthogonal matrix (N×N)
        ///flags
        ///    Operation flags; can be 0 or combination of the following:
        ///        * CV_SVD_MODIFY_A enables modification of matrix A during the operation. It speeds up the processing.
        ///        * CV_SVD_U_T means that the tranposed matrix U is returned. Specifying the flag speeds up the processing.
        ///        * CV_SVD_V_T means that the tranposed matrix V is returned. Specifying the flag speeds up the processing.
        ///
        /// It is the solution for homogeneous systems Ax=0,
        /// where x is the last column of the V matrix.
        ///
        /// M = U * S * V'
        /// M : [M x N] -> source, usually rectangular with M smaller N
        /// U : [M x M] -> Left Singular Vectors (in the Columns of U)
        /// V : [N x N] -> Right Singular Vectors (in the Columns of V)
        /// S : [M x N] -> Singular Values
        /// </summary>
        public static void ComputeSingularValues(this Matrix<double> A,
            out Matrix<double> UT, out Matrix<double> VT, out Matrix<double> S, bool thin)
        {
            ComputeSingularValuesUTSVT(A, out UT, out VT, out S, thin);
        }

        /// <summary>
        /// Note: the U and V matrices are TRANSPOSED!!! Transpose back by yourself if you need!
        ///
        ///A
        ///    Source M×N matrix.
        ///S
        ///    Resulting singular value matrix (M×N or N×N) or vector (N×1).
        ///U
        ///    Optional left orthogonal matrix (M×M or M×N). If CV_SVD_U_T is specified,
        ///    the number of rows and columns in the sentence above should be swapped.
        ///V
        ///    Optional right orthogonal matrix (N×N)
        ///flags
        ///    Operation flags; can be 0 or combination of the following:
        ///        * CV_SVD_MODIFY_A enables modification of matrix A during the operation. It speeds up the processing.
        ///        * CV_SVD_U_T means that the tranposed matrix U is returned. Specifying the flag speeds up the processing.
        ///        * CV_SVD_V_T means that the tranposed matrix V is returned. Specifying the flag speeds up the processing.
        ///
        /// It is the solution for homogeneous systems Ax=0,
        /// where x is the last column of the V matrix.
        ///
        /// M = U * S * V'
        /// M : [M x N] -> source, usually rectangular with M smaller N
        /// U : [M x M] -> Left Singular Vectors (in the Columns of U)
        /// V : [N x N] -> Right Singular Vectors (int the Columns)
        /// S : [M x N] -> Singular Values
        /// </summary>
        public static void ComputeSingularValues(this Matrix<double> A,
            out Matrix<double> UT, out Matrix<double> VT, out Matrix<double> S)
        {
            ComputeSingularValuesUTSVT(A, out UT, out VT, out S, false);
        }

        private static void ComputeSingularValuesUTSVT(this Matrix<double> A,
            out Matrix<double> UT, out Matrix<double> VT, out Matrix<double> S, bool thin)
        {
            //-- matrix must be of the form M<N, we check:
            if (A.SY < A.SX) throw new ArgumentException("Matrix must be of the form M>=N.");

            int M = (int)A.SY; // = Y = Rows = Width
            int N = (int)A.SX; // = X = Cols = Height

            int MM = M;
            if (thin) MM = N;

            S = new Matrix<double>(N, 1);
            UT = new Matrix<double>(M, MM);
            VT = new Matrix<double>(N, N);

            var mM = CvMat.FromPixelData(M, N, MatType.CV_64FC1, A.Array);
            var ms = CvMat.FromPixelData(N, 1, MatType.CV_64FC1, S.Array);
            var mu = CvMat.FromPixelData(M, MM, MatType.CV_64FC1, UT.Array);
            var mv = CvMat.FromPixelData(N, N, MatType.CV_64FC1, VT.Array);

            OpenCvSharp.SVD.Compute(mM, ms, mu, mv, OpenCvSharp.SVD.Flags.ModifyA);
        }

        private static void ComputeSingularValuesUTS(this Matrix<double> A,
            out Matrix<double> UT, out Matrix<double> S)
        {
            //-- matrix must be of the form M<N, we check:
            if (A.SY < A.SX) throw new ArgumentException("Matrix must be of the form M>=N.");

            int M = (int)A.SY; // = Y = Rows = Width
            int N = (int)A.SX; // = X = Cols = Height

            S = new Matrix<double>(N, 1);
            UT = new Matrix<double>(M, M);

            var mM = CvMat.FromPixelData(M, N, MatType.CV_64FC1, A.Array);
            var ms = CvMat.FromPixelData(N, 1, MatType.CV_64FC1, S.Array);
            var mu = CvMat.FromPixelData(M, M, MatType.CV_64FC1, UT.Array);
            var mv = new CvMat(N, N, MatType.CV_64FC1);

            OpenCvSharp.SVD.Compute(mM, ms, mu, mv, OpenCvSharp.SVD.Flags.ModifyA);
        }

        private static void ComputeSingularValuesVTS(this Matrix<double> A,
            out Matrix<double> VT, out Matrix<double> S)
        {
            //-- matrix must be of the form M<N, we check:
            if (A.SY < A.SX) throw new ArgumentException("Matrix must be of the form M>=N.");

            int M = (int)A.SY; // = Y = Rows = Width
            int N = (int)A.SX; // = X = Cols = Height

            S = new Matrix<double>(N, 1);
            VT = new Matrix<double>(N, N);

            var mM = CvMat.FromPixelData(M, N, MatType.CV_64FC1, A.Array);
            var ms = CvMat.FromPixelData(N, 1, MatType.CV_64FC1, S.Array);
            var mu = new CvMat(M, M, MatType.CV_64FC1);
            var mv = CvMat.FromPixelData(N, N, MatType.CV_64FC1, VT.Array);

            OpenCvSharp.SVD.Compute(mM, ms, mu, mv, OpenCvSharp.SVD.Flags.ModifyA);
        }

        /// <summary>
        /// Computes and returns the singularvalues and left and right singular-vectors of the given matrix.
        /// Note: the U and V matrice are TRANSPOSED!!! Transpose back by your self if you need!
        /// </summary>
        public static void ComputeSingularValues(this M44d M, out M44d U, out M44d V, out V4d S)
        {
            var s = new double[4 * 1];
            var u = new double[4 * 4];
            var v = new double[4 * 4];

            var pM = CvMat.FromPixelData(4, 4, MatType.CV_64FC1, M.ToArray());
            var ps = CvMat.FromPixelData(4, 1, MatType.CV_64FC1, s);
            var pu = CvMat.FromPixelData(4, 4, MatType.CV_64FC1, u);
            var pv = CvMat.FromPixelData(4, 4, MatType.CV_64FC1, v);

            OpenCvSharp.SVD.Compute(pM, ps, pu, pv, OpenCvSharp.SVD.Flags.ModifyA);

            U = new M44d(u).Transposed;
            V = new M44d(v);
            S = new V4d(s);
        }

        /// <summary>
        /// Computes and returns the (real) eigenvalues and eigenvectors of the given matrix.
        /// </summary>
        public static void ComputeEigenValues(this M44d M, out M44d eigenVectors, out V4d eigenValues)
        {
            var VV = new double[4 * 1];
            var EV = new double[4 * 4];

            int rows = 4;
            int cols = 4;

            var mat0 = CvMat.FromPixelData(rows, cols, MatType.CV_64FC1, M.ToArray());
            var mat1 = CvMat.FromPixelData(rows, 1, MatType.CV_64FC1, VV);
            var mat2 = CvMat.FromPixelData(rows, cols, MatType.CV_64FC1, EV);

            Cv2.Eigen(mat0, mat1, mat2);

            eigenValues = new V4d(VV);
            eigenVectors = new M44d(EV);
        }

        #endregion
    }
}
