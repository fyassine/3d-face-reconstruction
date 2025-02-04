#pragma once
#include <iostream>

class ProcrustesAligner {
public:
    Matrix4d estimatePose(const std::vector<Vector3d>& sourcePoints, const std::vector<Vector3d>& targetPoints) {
        ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");
        
        // We estimate the pose between source and target points using Procrustes algorithm.
        // Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
        // from source points to target points.
        
        auto sourceMean = computeMean(sourcePoints);
        auto targetMean = computeMean(targetPoints);
        
        Matrix3d rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
        Vector3d translation = computeTranslation(sourceMean, targetMean);
        Vector3d mockTranslation(0, 0, 0);
        float scale = computeScale(sourcePoints, targetPoints);
        
        // Compute the transformation matrix by using the computed rotation and translation.
        // You can access parts of the matrix with .block(start_row, start_col, num_rows, num_cols) = elements
        
        Matrix4d estimatedPose = Matrix4d::Identity();
                
        estimatedPose.block(0, 0, 3, 3) = scale * rotation;
        estimatedPose.block(0, 3, 3, 1) = targetMean - scale * rotation * sourceMean;

        return estimatedPose;
    }
    
private:
    Vector3d computeMean(const std::vector<Vector3d>& points) {
        // Compute the mean of input points.
        const unsigned nPoints = points.size();
        Vector3d mean = Vector3d::Zero();
        for (int i = 0; i < (int) nPoints; ++i) {
            mean += points[i];
        }
        mean /= nPoints;
        return mean;
    }

    Matrix3d estimateRotation(const std::vector<Vector3d>& sourcePoints, const Vector3d& sourceMean, const std::vector<Vector3d>& targetPoints, const Vector3d& targetMean) {
        // Estimate the rotation from source to target points, following the Procrustes algorithm.
        // To compute the singular value decomposition you can use JacobiSVD() from Eigen.
        const unsigned nPoints = sourcePoints.size();
        MatrixXd sourceMatrix(nPoints, 3);
        MatrixXd targetMatrix(nPoints, 3);

        for (int i = 0; i < (int) nPoints; ++i) {
            sourceMatrix.block(i, 0, 1, 3) = (sourcePoints[i] - sourceMean).transpose();
            targetMatrix.block(i, 0, 1, 3) = (targetPoints[i] - targetMean).transpose();
        }

        Matrix3d A = targetMatrix.transpose() * sourceMatrix;
        JacobiSVD<Matrix3d> svd(A, ComputeFullU | ComputeFullV);
        const Matrix3d& U = svd.matrixU();
        const Matrix3d& V = svd.matrixV();

        const double determinant = (U * V.transpose()).determinant();
        Matrix3d d_Matrix = Matrix3d::Identity();
        d_Matrix(2, 2) = determinant;

        Matrix3d rotation = U * d_Matrix * V.transpose();
        return rotation;
    }

    Vector3d computeTranslation(const Vector3d& sourceMean, const Vector3d& targetMean) {
        // Compute the translation vector from source to target points.
        Vector3d translation = targetMean - sourceMean;
        return translation;
    }
    
    double computeScale(const std::vector<Vector3d>& sourcePoints, const std::vector<Vector3d>& targetPoints) {
        ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same.");

        double sourceNormSum = 0.0;
        double targetNormSum = 0.0;

        // Compute means
        Vector3d sourceMean = computeMean(sourcePoints);
        Vector3d targetMean = computeMean(targetPoints);

        // Accumulate the squared distances for both source and target points (centered)
        for (size_t i = 0; i < sourcePoints.size(); ++i) {
            Vector3d centeredSource = sourcePoints[i] - sourceMean;
            Vector3d centeredTarget = targetPoints[i] - targetMean;

            sourceNormSum += centeredSource.squaredNorm();
            targetNormSum += centeredTarget.squaredNorm();
        }

        // Calculate the uniform scale as the ratio of sums
        return std::sqrt(targetNormSum / sourceNormSum);
    }
};
