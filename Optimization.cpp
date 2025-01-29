#include "Optimization.h"
#include <vector>
#include <iostream>
#include <chrono>

double GetDepthForVertex(Eigen::Vector3d vertex){
    return 0;
}

//Not sure about params and return type yet
void Optimization::optimizeDenseTerms(BfmProperties& properties, InputImage& inputImage) {

    ceres::Problem problemSparse;
    ceres::Problem problem;
    ceres::Problem problemColor;

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

    // Normalize normals
    for (size_t i = 0; i < bfmVertices.size(); i++) {
        normals[i].normalize();
    }

    int width = 1280;
    int height = 720;

    // Start Illumination
    std::ifstream inputFile(dataFolderPath + "face_39652.rps");
    json jsonData;
    inputFile >> jsonData;

    // Extract the RGB gamma coefficients from the "environmentMap" field
    auto coefficients = jsonData["environmentMap"]["coefficients"];

    // Create the Matrix3f and fill it with the extracted coefficients
    Matrix3f gammaMatrix;
    gammaMatrix(0, 0) = coefficients[0][0];
    gammaMatrix(0, 1) = coefficients[0][1];
    gammaMatrix(0, 2) = coefficients[0][2];

    gammaMatrix(1, 0) = coefficients[1][0];
    gammaMatrix(1, 1) = coefficients[1][1];
    gammaMatrix(1, 2) = coefficients[1][2];

    gammaMatrix(2, 0) = coefficients[2][0];
    gammaMatrix(2, 1) = coefficients[2][1];
    gammaMatrix(2, 2) = coefficients[2][2];
    
    
    std::vector<Eigen::Vector3f> illumination = std::vector<Vector3f>(bfmVertices.size(),
                                                                      Vector3f::Zero());
    for (size_t i = 0; i < bfmVertices.size(); i++) {
        illumination[i] = Illumination::computeIllumination(normals[i],
                                                            gammaMatrix);
    }
    // End illumination

    Eigen::VectorXd shapeParamsD = properties.shapeParams.cast<double>();
    Eigen::VectorXd expressionParamsD = properties.expressionParams.cast<double>();
    Eigen::VectorXd colorParamsD = properties.colorParams.cast<double>();

    /*auto landmarks_input_image = inputImage.landmarks;
    std::vector<Eigen::Vector2d> landmarks_bfm;
    for (const auto & landmark : properties.landmarks) {
        Eigen::Vector2f current_landmark = convert3Dto2D(landmark, inputImage.intrinsics, inputImage.extrinsics);
        landmarks_bfm.emplace_back(current_landmark);
    }*/

    auto landmarks_input_image = inputImage.landmarks;
    auto landmarks_depth_values = inputImage.depthValuesLandmarks;
    for (int i = 0; i < landmarks_input_image.size(); ++i) {
        auto current_landmark = convert2Dto3D(landmarks_input_image[i], landmarks_depth_values[i], inputImage.intrinsics, inputImage.extrinsics);
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimization, 3, 199, 100>(
                        new SparseOptimization(current_landmark, bfmVertices[properties.landmark_indices[i]], properties.landmark_indices[i], properties)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
    }

    /*auto landmarks_input_image = inputImage.landmarks;
    std::vector<Eigen::Vector2f> landmarks_depth_values;
    for (int i = 0; i < properties.landmark_indices.size(); ++i) {

    }
    for (int i = 0; i < landmarks_input_image.size(); ++i) {
        auto current_landmark = convert3Dto2D(bfmVertices[properties.landmark_indices[i]], inputImage.intrinsics, inputImage.extrinsics);
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimization, 2, 199, 100>(
                        new SparseOptimization(landmarks_input_image[i], bfmVertices[properties.landmark_indices[i]], properties.landmark_indices[i], properties, inputImage)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
    }*/

    std::cout << "Adding Residual Blocks: " << bfmVertices.size() << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point beginAdd = std::chrono::steady_clock::now();

    for (size_t i = 0; i < bfmVertices.size(); i+=25) {
        Eigen::Vector3f vertexBfm = bfmVertices[i];
        float depthInputImage = getDepthValueFromInputImage(vertexBfm, inputImage.depthValues, width, height, inputImage.intrinsics, inputImage.extrinsics);
        //Eigen::Vector3f colorInputImage = getColorValueFromInputImage(vertexBfm, inputImage.color, width, height, inputImage.intrinsics, inputImage.extrinsics);

        if(i == 0){
            std::chrono::steady_clock::time_point endAdd = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(endAdd - beginAdd).count() << "[µs]" << std::endl;
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (endAdd - beginAdd).count() << "[ns]" << std::endl;
        }
        
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 1, 199, 100>(
                        new GeometryOptimization(bfmVertices[i], depthInputImage, normals[i], properties, i)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
        
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimizationPointToPlane, 1, 199, 100>(
                        new GeometryOptimizationPointToPlane(bfmVertices[i], depthInputImage, normals[i], properties, i)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );

        if(i == 0){
            std::chrono::steady_clock::time_point endAdd = std::chrono::steady_clock::now();
            std::cout << "Time difference Add Res Block = " << std::chrono::duration_cast<std::chrono::microseconds>(endAdd - beginAdd).count() << "[µs]" << std::endl;
            std::cout << "Time difference Add Res Block = " << std::chrono::duration_cast<std::chrono::nanoseconds> (endAdd - beginAdd).count() << "[ns]" << std::endl;
        }
//    }

    //TODO: Color
        /*Eigen::Vector3f colorInputImage = getColorValueFromInputImage(vertexBfm, inputImage.color, width, height, inputImage.intrinsics, inputImage.extrinsics);
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ColorOptimization, 1, 199>(
                        new ColorOptimization(bfmColors[i], colorInputImage, illumination[i], properties, i)
                ),
                nullptr,
                colorParamsD.data()
        );*/
    }

    std::cout << "Stop" << std::endl;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;

    // Initialize standard deviation vectors from PCA variance
    std::vector<double> identity_std_dev(199);
    std::vector<double> albedo_std_dev(199);
    std::vector<double> expression_std_dev(100);

    // Fill identity and albedo standard deviations
    for (int i = 0; i < 199; ++i) {
        identity_std_dev[i] = std::sqrt(properties.shapePcaVariance[i]);
        albedo_std_dev[i] = std::sqrt(properties.colorPcaVariance[i]);
    }

    // Fill expression standard deviations
    for (int i = 0; i < 100; ++i) {
        expression_std_dev[i] = std::sqrt(properties.expressionPcaVariance[i]);
    }

    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GeometryRegularizationTerm, 2, 199, 100>(
                    new GeometryRegularizationTerm(identity_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            expressionParamsD.data()
    );

    /*problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ColorRegularizationTerm, 1, 199>(
                    new ColorRegularizationTerm(albedo_std_dev)),
            nullptr,
            colorParamsD.data()
    );*/

    // Setup and run solver
    std::cout << "\n=== Configuring Solver ===\n";
    ceres::Solver::Options options;
    ceres::Solver::Options optionsSparse;
    configureSolver(options);
    configureSolver(optionsSparse);
    optionsSparse.max_num_iterations = 1000;
    ceres::Solver::Summary summary;
    ceres::Solver::Summary summarySparse;
    ceres::Solver::Summary summaryColor;

    std::cout << "\n=== Starting Optimization ===\n";
    //ceres::Solve(optionsSparse, &problemSparse, &summarySparse);
    ceres::Solve(options, &problem, &summary);
    //ceres::Solve(options, &problemColor, &summaryColor);

    Eigen::IOFormat CleanFmt(4, 0, ", ", " ", "[", "]");
    std::cout << "\n=== Optimization Results ===\n";
    std::cout << "Initial cost: " << summary.initial_cost << std::endl;
    std::cout << "Final cost: " << summary.final_cost << std::endl;
    std::cout << "Number of iterations: " << summary.iterations.size() << std::endl;
    std::cout << "Termination type: " << summary.termination_type << std::endl;
    std::cout << "Termination message: " << summary.message << std::endl;

    std::cout << "\nShape and expression params after optimization:" << std::endl;
    std::cout << "Shape Params: " << shapeParamsD.format(CleanFmt) << std::endl;
    std::cout << "Expression Params: " << expressionParamsD.format(CleanFmt) << std::endl;
    std::cout << "Color Params: " << colorParamsD.format(CleanFmt) << std::endl;

    properties.shapeParams = shapeParamsD.cast<float>();
    properties.expressionParams = expressionParamsD.cast<float>();
    properties.colorParams = colorParamsD.cast<float>();
}

void Optimization::optimizeSparseTerms() {
    ceres::Problem problem;
}

void Optimization::configureSolver(ceres::Solver::Options &options) {
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = ceres::DENSE_QR;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = 1;
    options.max_num_iterations = 1000;
    options.num_threads = 12;
}





void Optimization::optimize(BfmProperties& bfm, InputImage& inputImage) {
    optimizeSparseTerms();
    optimizeDenseTerms(bfm, inputImage);
    //regularize(bfm);
}



/*void Optimization::optimizeDenseTerms(BfmProperties& bfm, InputImage& inputImage) {

    ceres::Problem problem;
    auto bfmVertices = getVertices(bfm);
    auto normals = getNormals(bfm);

    Eigen::VectorXd shapeParamsD = bfm.shapeParams.cast<double>();
    Eigen::VectorXd expressionParamsD = bfm.expressionParams.cast<double>();
    Eigen::VectorXd colorParamsD = bfm.colorParams.cast<double>();

    for (size_t i = 0; i < bfmVertices.size(); ++i) {
        auto vertexBfm = bfmVertices[i];
        int width = 1280;
        int height = 720;
        float depthInputImage = getDepthValueFromInputImage(vertexBfm, inputImage.depthValues, width, height, inputImage.intrinsics, inputImage.extrinsics);

        auto geometryFunction = new ceres::AutoDiffCostFunction<GeometryOptimization, 2, 199, 100>(
                new GeometryOptimization(bfmVertices[i], depthInputImage, normals[i], bfm.shapePcaBasis, i)
        );

        problem.AddResidualBlock(
                geometryFunction,
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
    }
    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;
    std::cout << "\n=== Starting Optimization ===\n";
    ceres::Solve(options, &problem, &summary);

    bfm.shapeParams = shapeParamsD.cast<float>();
    bfm.expressionParams = expressionParamsD.cast<float>();
    bfm.colorParams = colorParamsD.cast<float>();
}*/

void Optimization::regularize(BfmProperties& bfm) {

    ceres::Problem problem;

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

    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationTerm, 1, 199, 199, 100>(
                    new RegularizationTerm(identity_std_dev, albedo_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            colorParamsD.data(),
            expressionParamsD.data()
    );

    /*problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationFunction, 199, 199>(
                    new RegularizationFunction(5, 199)),
            nullptr,
            shapeParamsD.data()
    );

    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationFunction, 100, 100>(
                    new RegularizationFunction(5, 100)),
            nullptr,
            expressionParamsD.data()
    );*/
    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;
    std::cout << "\n=== Starting Regularization ===\n";
    ceres::Solve(options, &problem, &summary);

    bfm.shapeParams = shapeParamsD.cast<float>();
    bfm.expressionParams = expressionParamsD.cast<float>();
    bfm.colorParams = colorParamsD.cast<float>();
}

