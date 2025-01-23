#include "Optimization.h"
#include <vector>
#include <iostream>

double GetDepthForVertex(Eigen::Vector3d vertex){
    return 0;
}

//Not sure about params and return type yet
void Optimization::optimizeDenseTerms(BfmProperties& properties, InputImage& inputImage) {
    std::cout << "\n=== Starting Dense Terms Optimization ===\n";

    ceres::Problem problem;
    ceres::Problem problemColor;
    // Debug BFM vertices
    auto bfmVertices = getVertices(properties);
    auto bfmColors = getColorValuesF(properties);
    std::cout << "Number of BFM vertices: " << bfmVertices.size() << std::endl;
    std::cout << "First vertex position: ("
              << bfmVertices[0].x() << ", "
              << bfmVertices[0].y() << ", "
              << bfmVertices[0].z() << ")\n";

    // Debug normals calculation
    std::cout << "\n=== Computing Normals ===\n";
    std::vector<Vector3f> normals = std::vector<Vector3f>(bfmVertices.size(), Vector3f::Zero());
    int validNormalCount = 0;

    for (size_t i = 0; i < properties.triangles.size(); i+=3) {
        auto triangle0 = properties.triangles[i];
        auto triangle1 = properties.triangles[i+1];
        auto triangle2 = properties.triangles[i+2];

        // Debug triangle indices
        if (i == 0) {
            std::cout << "First triangle indices: " << triangle0 << ", " << triangle1 << ", " << triangle2 << std::endl;
        }

        Vector3f faceNormal = (bfmVertices[triangle1] - bfmVertices[triangle0]).cross(bfmVertices[triangle2] - bfmVertices[triangle0]);

        if (faceNormal.norm() > 0) {
            validNormalCount++;
        }

        normals[triangle0] += faceNormal;
        normals[triangle1] += faceNormal;
        normals[triangle2] += faceNormal;
    }

    std::cout << "Valid face normals computed: " << validNormalCount << std::endl;

    // Normalize normals
    int zeroNormals = 0;
    for (size_t i = 0; i < bfmVertices.size(); i++) {
        if (normals[i].norm() < 1e-10) {
            zeroNormals++;
        }
        normals[i].normalize();
    }
    std::cout << "Number of zero-length normals: " << zeroNormals << std::endl;

    int width = 1280;
    int height = 720;

    // Start Illumination
    std::ifstream inputFile(dataFolderPath + "face_39736.rps");
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
        
    // Debug parameters
    std::cout << "\n=== Parameters Setup ===\n";
    Eigen::VectorXd shapeParamsD = properties.shapeParams.cast<double>();
    Eigen::VectorXd expressionParamsD = properties.expressionParams.cast<double>();
    Eigen::VectorXd colorParamsD = properties.colorParams.cast<double>();

    std::cout << "Shape params size: " << shapeParamsD.size() << std::endl;
    std::cout << "Expression params size: " << expressionParamsD.size() << std::endl;

    // Add residual blocks
    std::cout << "\n=== Adding Residual Blocks ===\n";
    int validResiduals = 0;
    int invalidDepthValues = 0;

    for (size_t i = 0; i < bfmVertices.size(); ++i) {
        Eigen::Vector3f vertexBfm = bfmVertices[i];
        float depthInputImage = getDepthValueFromInputImage(vertexBfm, inputImage.depthValues, width, height, inputImage.intrinsics, inputImage.extrinsics);
        Eigen::Vector3f colorInputImage = getColorValueFromInputImage(vertexBfm, inputImage.color, width, height, inputImage.intrinsics, inputImage.extrinsics);

        if (std::isnan(depthInputImage) || std::isinf(depthInputImage)) {
            invalidDepthValues++;
            continue;
        }

        // Debug first few vertices
        if (i < 5) {
            std::cout << "Vertex " << i << " depth: " << depthInputImage << std::endl;
            std::cout << "Vertex " << i << " color: " << colorInputImage << std::endl;
            std::cout << "Vertex " << i << " normal: ("
                      << normals[i].x() << ", "
                      << normals[i].y() << ", "
                      << normals[i].z() << ")\n";
        }

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 2, 199, 100>(
                        new GeometryOptimization(bfmVertices[i], depthInputImage, normals[i], properties.shapePcaBasis, properties.expressionPcaBasis, i)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
        );

        problemColor.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ColorOptimization, 1, 199>(
                        new ColorOptimization(bfmColors[i], colorInputImage, illumination[i], properties.colorPcaBasis, i)
                ),
                nullptr,
                colorParamsD.data()
        );
        validResiduals++;
    }

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

    /*problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationTerm, 1, 199, 199, 100>(
                    new RegularizationTerm(identity_std_dev, albedo_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            colorParamsD.data(),
            expressionParamsD.data()
    );*/

    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GeometryRegularizationTerm, 1, 199, 100>(
                    new GeometryRegularizationTerm(identity_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            expressionParamsD.data()
    );

    problemColor.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ColorRegularizationTerm, 1, 199>(
                    new ColorRegularizationTerm(albedo_std_dev)),
            nullptr,
            colorParamsD.data()
    );


    std::cout << "Valid residual blocks added: " << validResiduals << std::endl;
    std::cout << "Invalid depth values encountered: " << invalidDepthValues << std::endl;

    // Setup and run solver
    std::cout << "\n=== Configuring Solver ===\n";
    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;
    ceres::Solver::Summary summaryColor;

    std::cout << "\n=== Starting Optimization ===\n";
    //ceres::Solve(options, &problem, &summary);
    ceres::Solve(options, &problemColor, &summaryColor);

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

    //TODO: Print color to see change

    std::cout << "\nChange in Color: " << std::endl;
    auto colorResult = getColorValuesF(properties);
    for (int i = 0; i < 20; ++i) {
        auto currentColor = colorResult[i];
        std::cout << "(" << currentColor.x() << ", " << currentColor.y() << ", " << currentColor.z() << ")" << std::endl;
    }
    properties.shapeParams = shapeParamsD.cast<float>();
    properties.expressionParams = expressionParamsD.cast<float>();
    properties.colorParams = colorParamsD.cast<float>();

    std::cout << "Color after optimization" << std::endl;
    auto colorResultAfterOpt = getColorValuesF(properties);
    for (int i = 0; i < 20; ++i) {
        auto currentColor = colorResultAfterOpt[i];
        std::cout << "(" << currentColor.x() << ", " << currentColor.y() << ", " << currentColor.z() << ")" << std::endl;
    }
    float smallestValue = 0;
    float biggestValue = -1000.0f;
    for (int i = 0; i < colorResultAfterOpt.size(); ++i) {
        auto currentColor = colorResultAfterOpt[i];
        if(currentColor.x() < smallestValue){
            smallestValue = currentColor.x();
        }
        if(currentColor.y() < smallestValue){
            smallestValue = currentColor.y();
        }
        if(currentColor.z() < smallestValue){
            smallestValue = currentColor.z();
        }

        if(currentColor.x() > biggestValue){
            biggestValue = currentColor.x();
        }
        if(currentColor.y() > biggestValue){
            biggestValue = currentColor.y();
        }
        if(currentColor.z() > biggestValue){
            biggestValue = currentColor.z();
        }
    }

    std::cout << "Smallest Value: " << smallestValue << std::endl;
    std::cout << "Biggest Value: " << biggestValue << std::endl;
}

void Optimization::optimizeSparseTerms() {
    ceres::Problem problem;
}

void Optimization::configureSolver(ceres::Solver::Options &options) {
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    //options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = 1;
    options.max_num_iterations = 50; //maybe make it 100
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

