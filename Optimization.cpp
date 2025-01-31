#include "Optimization.h"
#include <vector>
#include <iostream>
#include <chrono>

void Optimization::optimizeDenseTerms(BfmProperties& properties, InputImage& inputImage, ceres::Problem& problem) {

    auto bfmVertices = getVertices(properties);
    auto bfmColors = getColorValuesF(properties);

    std::vector<Vector3f> normals = std::vector<Vector3f>(bfmVertices.size(), Vector3f::Zero());

    for (size_t i = 0; i < properties.triangles.size(); i+=3) {
        auto triangle0 = properties.triangles[i];
        auto triangle1 = properties.triangles[i+1];
        auto triangle2 = properties.triangles[i+2];

        Vector3f faceNormal = (bfmVertices[triangle1] - bfmVertices[triangle0]).cross(bfmVertices[triangle2] - bfmVertices[triangle0]);

        normals[triangle0] += faceNormal;
        normals[triangle1] += faceNormal;
        normals[triangle2] += faceNormal;
    }

    for (size_t i = 0; i < bfmVertices.size(); i++) {
        normals[i].normalize();
    }

    int width = 1280;
    int height = 720;

    // Start Illumination

    Eigen::VectorXd shapeParamsD = properties.shapeParams.cast<double>();
    Eigen::VectorXd expressionParamsD = properties.expressionParams.cast<double>();
    Eigen::VectorXd colorParamsD = properties.colorParams.cast<double>();

    for (int i = 0; i < bfmVertices.size(); i+=100) {
        Eigen::Vector3f vertexBfm = bfmVertices[i];
        float depthInputImage = getDepthValueFromInputImage(vertexBfm, inputImage.depthValues, width, height, inputImage.intrinsics, inputImage.extrinsics);
        
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 2, 199, 100>(
                        new GeometryOptimization(bfmVertices[i], depthInputImage, normals[i], properties, i)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
    }

    properties.shapeParams = shapeParamsD.cast<float>();
    properties.expressionParams = expressionParamsD.cast<float>();
    properties.colorParams = colorParamsD.cast<float>();
}

/*void Optimization::optimizeSparseTerms(BfmProperties& bfm, InputImage& inputImage, Eigen::VectorXd& shapeParamsD, Eigen::VectorXd& expressionParamsD) {
    ceres::Problem problem;
    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;
    auto bfmVertices = getVertices(bfm);
    auto landmarks_input_image = inputImage.landmarks;
    auto landmarks_depth_values = inputImage.depthValuesLandmarks;
    for (int i = 0; i < landmarks_input_image.size(); ++i) {
        auto current_landmark = convert2Dto3D(landmarks_input_image[i], landmarks_depth_values[i], inputImage.intrinsics, inputImage.extrinsics);
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimization, 3, 199, 100>(
                        new SparseOptimization(current_landmark, bfmVertices[bfm.landmark_indices[i]], bfm.landmark_indices[i], bfm)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
    }
}*/

void Optimization::optimize(BfmProperties& bfm, InputImage& inputImage) {
    std::cout << "Start Optimization" << std::endl;
    Eigen::VectorXd shapeParamsD = bfm.shapeParams.cast<double>();
    Eigen::VectorXd expressionParamsD = bfm.expressionParams.cast<double>(); //maybe thats the problem? What about creating params as std::vector?!
    Eigen::VectorXd colorParamsD = bfm.colorParams.cast<double>();
    ceres::Problem sparseProblem;
    ceres::Problem problem;
    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;
    ceres::Solver::Summary sparseSummary;

    auto bfmVertices = getVerticesWithoutProcrustes(bfm);
    convertVerticesTest(bfmVertices, resultFolderPath + "DaskönntederFehlersein.ply");
    auto landmarks_input_image = inputImage.landmarks;
    auto landmarks_depth_values = inputImage.depthValuesLandmarks;

    std::vector<double> offsets;
    for (int i = 0; i < 204; ++i) {
        offsets.emplace_back(0);
    }

    std::cout << landmarks_input_image.size() << std::endl;
    for (int i = 0; i < landmarks_input_image.size(); ++i) {
        auto current_landmark = convert2Dto3D(landmarks_input_image[i], landmarks_depth_values[i], inputImage.intrinsics, inputImage.extrinsics);
        std::cout << "Current Input Landmarks(" << i << "): " << std::endl;
        std::cout << current_landmark.cast<double>() << std::endl;
        std::cout << "Current BFM Vertices(" << i << "): " << std::endl;
        std::cout << bfmVertices[bfm.landmark_indices[i]].cast<double>() << std::endl;
        //TODO: Transform point manually
        sparseProblem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimization, 3, 199, 100, 204>(
                        new SparseOptimization(current_landmark.cast<double>(), bfmVertices[bfm.landmark_indices[i]].cast<double>(), bfm.landmark_indices[i], bfm, i)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data(),
                offsets.data()
        );
    }

    std::cout << "Current Landmarks End" << std::endl;
    std::vector<double> identity_std_dev(199);
    std::vector<double> albedo_std_dev(199);
    std::vector<double> expression_std_dev(100);
    // Fill identity and albedo standard deviations
    for (int i = 0; i < 199; ++i) {
        identity_std_dev[i] = std::sqrt(bfm.shapePcaVariance[i]);
        albedo_std_dev[i] = std::sqrt(bfm.colorPcaVariance[i]);
    }
    // Fill expression standard deviations
    for (int i = 0; i < 100; ++i) {
        expression_std_dev[i] = std::sqrt(bfm.expressionPcaVariance[i]);
    }
    /*sparseProblem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GeometryRegularizationTerm, 2, 199, 100>(
                    new GeometryRegularizationTerm(identity_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            expressionParamsD.data()
    );*/
    //optimizeDenseTerms(bfm, inputImage, problem);
    //regularize(bfm, problem);

    ceres::Solve(options, &sparseProblem, &sparseSummary);
    for (int i = 0; i < offsets.size(); i += 3) {
        auto current_landmark = convert2Dto3D(landmarks_input_image[i/3], landmarks_depth_values[i/3], inputImage.intrinsics, inputImage.extrinsics);

        std::cout << "Offset Vector: " << offsets[i] << ", " << offsets[i + 1] << ", " << offsets[i + 2] << std::endl;
        std::cout << "Image Landmark: " << current_landmark.x() << ", " << current_landmark.y() << ", " << current_landmark.z() << std::endl;
        std::cout << "BFM Landmark: " << bfmVertices[bfm.landmark_indices[i/3]].x() << ", " << bfmVertices[bfm.landmark_indices[i/3]].y() << ", " << bfmVertices[bfm.landmark_indices[i/3]].z() << std::endl;
        auto result = Eigen::Vector4f(bfmVertices[bfm.landmark_indices[i/3]].x(), bfmVertices[bfm.landmark_indices[i/3]].y(), bfmVertices[bfm.landmark_indices[i/3]].z(), 1.0f);
        auto transformedLandmark = bfm.transformation * result;
        std::cout << "BFM Landmark + Offset: " << transformedLandmark.x() + offsets[i] << ", " << transformedLandmark.y() + offsets[i + 1] << ", " << transformedLandmark.z() + offsets[i+2] << std::endl;
    }
    //print targetlandmarks, offsets
    //TODO Dense: GETVERTICESWITHOUT PROCRUSTES erneut callen -> because new vertices are important
    auto bfmVerticesDepth = getVertices(bfm);
    std::vector<Vector3f> normals = std::vector<Vector3f>(bfmVertices.size(), Vector3f::Zero());

    for (size_t i = 0; i < bfm.triangles.size(); i+=3) {
        auto triangle0 = bfm.triangles[i];
        auto triangle1 = bfm.triangles[i+1];
        auto triangle2 = bfm.triangles[i+2];

        Vector3f faceNormal = (bfmVertices[triangle1] - bfmVertices[triangle0]).cross(bfmVertices[triangle2] - bfmVertices[triangle0]);

        normals[triangle0] += faceNormal;
        normals[triangle1] += faceNormal;
        normals[triangle2] += faceNormal;
    }

    for (size_t i = 0; i < bfmVertices.size(); i++) {
        normals[i].normalize();
    }

    int width = 1280;
    int height = 720;
    std::cout << "Start: " << bfmVertices.size() << std::endl;

    for (int i = 0; i < bfmVertices.size(); i+=100) {
        Eigen::Vector3f vertexBfm = bfmVerticesDepth[i];
        std::cout << i << ": " << bfmVerticesDepth[i] << std::endl;
        float depthInputImage = getDepthValueFromInputImage(vertexBfm, inputImage.depthValues, width, height, inputImage.intrinsics, inputImage.extrinsics);
        std::cout << "Depth: " << ": " << depthInputImage << std::endl;
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 2, 199, 100>(
                        new GeometryOptimization(bfmVertices[i], depthInputImage, normals[i], bfm, i)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
    }
    std::cout << "End" << std::endl;


    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GeometryRegularizationTerm, 2, 199, 100>(
                    new GeometryRegularizationTerm(identity_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            expressionParamsD.data()
    );

    //options.max_num_iterations = 100;
    ceres::Solve(options, &problem, &summary);
    bfm.shapeParams = shapeParamsD.cast<float>();
    bfm.expressionParams = expressionParamsD.cast<float>();
    bfm.colorParams = colorParamsD.cast<float>();
}

void Optimization::regularize(BfmProperties& bfm, ceres::Problem& problem) {
    Eigen::VectorXd shapeParamsD = bfm.shapeParams.cast<double>();
    Eigen::VectorXd expressionParamsD = bfm.expressionParams.cast<double>();
    Eigen::VectorXd colorParamsD = bfm.colorParams.cast<double>();
    // Initialize standard deviation vectors from PCA variance
    std::vector<double> identity_std_dev(199);
    std::vector<double> albedo_std_dev(199);
    std::vector<double> expression_std_dev(100);
    // Fill identity and albedo standard deviations
    for (int i = 0; i < 199; ++i) {
        identity_std_dev[i] = std::sqrt(bfm.shapePcaVariance[i]);
        albedo_std_dev[i] = std::sqrt(bfm.colorPcaVariance[i]);
    }
    // Fill expression standard deviations
    for (int i = 0; i < 100; ++i) {
        expression_std_dev[i] = std::sqrt(bfm.expressionPcaVariance[i]);
    }
    /*problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GeometryRegularizationTerm, 2, 199, 100>(
                    new GeometryRegularizationTerm(identity_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            expressionParamsD.data()
    );*/
    bfm.shapeParams = shapeParamsD.cast<float>();
    bfm.expressionParams = expressionParamsD.cast<float>();
    bfm.colorParams = colorParamsD.cast<float>();
}

void Optimization::configureSolver(ceres::Solver::Options &options) {
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = 1;
    options.max_num_iterations = 2000;
    options.num_threads = 24;
}

