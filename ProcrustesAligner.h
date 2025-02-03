#pragma once
#include <iostream>

class ProcrustesAligner {
public:
    Matrix4f estimatePose(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
        ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");
        
        // We estimate the pose between source and target points using Procrustes algorithm.
        // Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
        // from source points to target points.
        
        auto sourceMean = computeMean(sourcePoints);
        auto targetMean = computeMean(targetPoints);
        
        Matrix3f rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
        Vector3f translation = computeTranslation(sourceMean, targetMean);
        Vector3f mockTranslation(0, 0, 0);
        float scale = computeScale(sourcePoints, targetPoints);
        
        // Compute the transformation matrix by using the computed rotation and translation.
        // You can access parts of the matrix with .block(start_row, start_col, num_rows, num_cols) = elements
        
        Matrix4f estimatedPose = Matrix4f::Identity();
                
        estimatedPose.block(0, 0, 3, 3) = scale * rotation;
        estimatedPose.block(0, 3, 3, 1) = targetMean - scale * rotation * sourceMean;

        return estimatedPose;
    }
    
private:
    Vector3f computeMean(const std::vector<Vector3f>& points) {
        // Compute the mean of input points.
        const unsigned nPoints = points.size();
        Vector3f mean = Vector3f::Zero();
        for (int i = 0; i < (int) nPoints; ++i) {
            mean += points[i];
        }
        mean /= nPoints;
        return mean;
    }

    Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
        // Estimate the rotation from source to target points, following the Procrustes algorithm.
        // To compute the singular value decomposition you can use JacobiSVD() from Eigen.
        const unsigned nPoints = sourcePoints.size();
        MatrixXf sourceMatrix(nPoints, 3);
        MatrixXf targetMatrix(nPoints, 3);

        for (int i = 0; i < (int) nPoints; ++i) {
            sourceMatrix.block(i, 0, 1, 3) = (sourcePoints[i] - sourceMean).transpose();
            targetMatrix.block(i, 0, 1, 3) = (targetPoints[i] - targetMean).transpose();
        }

        Matrix3f A = targetMatrix.transpose() * sourceMatrix;
        JacobiSVD<Matrix3f> svd(A, ComputeFullU | ComputeFullV);
        const Matrix3f& U = svd.matrixU();
        const Matrix3f& V = svd.matrixV();

        const float determinant = (U * V.transpose()).determinant();
        Matrix3f d_Matrix = Matrix3f::Identity();
        d_Matrix(2, 2) = determinant;

        Matrix3f rotation = U * d_Matrix * V.transpose();
        return rotation;
    }

    Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean) {
        // Compute the translation vector from source to target points.
        Vector3f translation = targetMean - sourceMean;
        return translation;
    }
    
    float computeScale(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
        ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same.");

        float sourceNormSum = 0.0f;
        float targetNormSum = 0.0f;

        // Compute means
        Vector3f sourceMean = computeMean(sourcePoints);
        Vector3f targetMean = computeMean(targetPoints);

        // Accumulate the squared distances for both source and target points (centered)
        for (size_t i = 0; i < sourcePoints.size(); ++i) {
            Vector3f centeredSource = sourcePoints[i] - sourceMean;
            Vector3f centeredTarget = targetPoints[i] - targetMean;

            sourceNormSum += centeredSource.squaredNorm();
            targetNormSum += centeredTarget.squaredNorm();
        }

        // Calculate the uniform scale as the ratio of sums
        return std::sqrt(targetNormSum / sourceNormSum);
    }
};
