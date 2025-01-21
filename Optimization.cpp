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

    // Debug BFM vertices
    auto bfmVertices = getVertices(properties);
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

        if (std::isnan(depthInputImage) || std::isinf(depthInputImage)) {
            invalidDepthValues++;
            continue;
        }

        // Debug first few vertices
        if (i < 5) {
            std::cout << "Vertex " << i << " depth: " << depthInputImage << std::endl;
            std::cout << "Vertex " << i << " normal: ("
                      << normals[i].x() << ", "
                      << normals[i].y() << ", "
                      << normals[i].z() << ")\n";
        }

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 2, 199, 100>(
                        new GeometryOptimization(bfmVertices[i], depthInputImage, normals[i], properties.shapePcaBasis, i)
                ),
                nullptr,
                shapeParamsD.data(),
                expressionParamsD.data()
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

    problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationTerm, 1, 199, 199, 100>(
                    new RegularizationTerm(identity_std_dev, albedo_std_dev, expression_std_dev)),
            nullptr,
            shapeParamsD.data(),
            colorParamsD.data(),
            expressionParamsD.data()
    );

    std::cout << "Valid residual blocks added: " << validResiduals << std::endl;
    std::cout << "Invalid depth values encountered: " << invalidDepthValues << std::endl;

    // Setup and run solver
    std::cout << "\n=== Configuring Solver ===\n";
    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;

    std::cout << "\n=== Starting Optimization ===\n";
    ceres::Solve(options, &problem, &summary);

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
    options.max_num_iterations = 5; //maybe make it 100
    options.num_threads = 12;
}


