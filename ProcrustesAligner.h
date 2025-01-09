#pragma once
#include "SimpleMesh.h"

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
        Vector3f translation = computeTranslation(sourceMean, targetMean, rotation);
        float scale = computeScale(sourcePoints, targetPoints, rotation);
        
        std::cout << "rotation:" << std::endl;
        std::cout << rotation << std::endl;
        
        std::cout << "translation:" << std::endl;
        std::cout << translation << std::endl;
        
        std::cout << "scale:" << scale << std::endl;
        
        // Compute the transformation matrix by using the computed rotation and translation.
        // You can access parts of the matrix with .block(start_row, start_col, num_rows, num_cols) = elements
        
        Matrix4f estimatedPose = Matrix4f::Identity();
                
        estimatedPose.block(0, 0, 3, 3) = scale * rotation;
        estimatedPose.block(0, 3, 3, 1) = translation;
        
        std::cout << "estimatedPose:" << std::endl;
        std::cout << estimatedPose << std::endl;

        return estimatedPose;
    }
    
private:
    Vector3f computeMean(const std::vector<Vector3f>& points) {
        // Compute the mean of input points.
        // Hint: You can use the .size() method to get the length of a vector.
        
        Vector3f mean = Vector3f::Zero();
        
        for (unsigned int i=0; i<points.size(); i++) {
            mean.x() = (mean.x() + points[i].x())/(i+1);
            mean.y() = (mean.y() + points[i].y())/(i+1);
            mean.z() = (mean.z() + points[i].z())/(i+1);
        }
                
        return mean;
    }
    
    Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
        // Estimate the rotation from source to target points, following the Procrustes algorithm.
        // To compute the singular value decomposition you can use JacobiSVD() from Eigen.
        // Hint: You can initialize an Eigen matrix with "MatrixXf m(num_rows,num_cols);" and access/modify parts of it using the .block() method (see above).
        
        Matrix3f rotation = Matrix3f::Identity();
        
        unsigned long pointsSize = sourcePoints.size();
        
        // mean-centered points
        MatrixXf sourceCentered(3, pointsSize);
        MatrixXf targetCentered(3, pointsSize);
        
        for (size_t i = 0; i < pointsSize; ++i) {
            sourceCentered.block(0, i, 3, 1) = sourcePoints[i] - sourceMean;
            targetCentered.block(0, i, 3, 1) = targetPoints[i] - targetMean;
        }
        
        // covariance matrix
        MatrixXf cov = targetCentered * sourceCentered.transpose();

        // SVD
        JacobiSVD<MatrixXf> svd(cov, ComputeThinU | ComputeThinV);
        MatrixXf U = svd.matrixU();
        MatrixXf V = svd.matrixV();

        rotation = U * V.transpose();

        MatrixXf identity = MatrixXf::Identity(V.rows(), V.cols());

        if (rotation.determinant() < 0) {
            identity(V.cols() - 1, V.cols() - 1) = -1;
            rotation = V * identity * U.transpose();
        }
        
        return rotation;
    }
    
    Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean, const Matrix3f& rotation) {
        // Compute the translation vector from source to target points.
        
        Vector3f translation = Vector3f::Zero();
        
        translation = targetMean - rotation * sourceMean;
        
        return translation;
    }
    
    float computeScale(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const Matrix3f& rotation) {
        ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same.");

        float sourceNormSum = 0.0f;
        float targetNormSum = 0.0f;
        
        // Accumulate the squared distances for both source and target points
        for (size_t i = 0; i < sourcePoints.size(); ++i) {
            Vector3f rotatedSource = rotation * sourcePoints[i];  // Rotate the source point
            sourceNormSum += rotatedSource.squaredNorm();
            targetNormSum += targetPoints[i].squaredNorm();
        }
        
        // Calculate the uniform scale as the ratio of sums
        return std::sqrt(targetNormSum / sourceNormSum);
    }
};
