#include "Optimization.h"
#include <vector>
#include <iostream>

double GetDepthForVertex(Eigen::Vector3d vertex){
    return 0;
}

//Not sure about params and return type yet
void Optimization::optimizeDenseTerms(BfmProperties& properties, InputImage& inputImage) {
    ceres::Problem problem;

    // Mock data
    std::vector<Eigen::Vector3f> vertices; // Fill with actual data


    std::vector<Eigen::Vector3d> rgbData;
    std::vector<double> depths;
    std::vector<Eigen::Vector3f> normals;
    auto bfmVertices = getVertices(properties);

    for (int i = 0; i < bfmVertices.size(); ++i) {
        normals.emplace_back(1, 0, 0);
    }
    int width = 1280;
    int height = 720;

    //End Mock data
    Eigen::VectorXd shapeParamsD = properties.shapeParams.cast<double>();
    Eigen::VectorXd expressionParamsD = properties.expressionParams.cast<double>();

    std::cout << "Heyy" << std::endl;
    for (size_t i = 0; i < bfmVertices.size(); ++i) {
        Eigen::Vector3f vertexBfm = bfmVertices[i];
        float depthInputImage = getDepthValueFromInputImage(vertexBfm, inputImage.depthValues, width, height, inputImage.intrinsics, inputImage.extrinsics);

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 2, 100, 199>(
                        new GeometryOptimization(bfmVertices[i], depthInputImage, normals[i])
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );
        /*problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ColorOptimization, 1, 3>(
                new ColorOptimization(rgbData[i], rgbData[i])), // replace second rgbData[i] with illumination
                nullptr,
                colorParams.data()
                );*/
    }

    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    properties.shapeParams = shapeParamsD.cast<float>();
    properties.expressionParams = expressionParamsD.cast<float>();
}

void Optimization::optimizeSparseTerms() {
    ceres::Problem problem;
}

void Optimization::configureSolver(ceres::Solver::Options &options) {
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = 1;
    options.max_num_iterations = 1; //maybe make it 100
    options.num_threads = 8;
}


