#include "Optimizer.h"
#include "Illumination.h"
#include <random>
#include <chrono>

Optimizer::Optimizer(BaselFaceModel *baselFaceModel, InputData *inputData) {
    m_baselFaceModel = baselFaceModel;
    m_inputData = inputData;
    configureSolver();
}

Optimizer::~Optimizer() = default;

void Optimizer::optimizeSparseTerms() {
    ceres::Problem problem;

    auto landmark_indices_bfm = m_baselFaceModel->getLandmarkIndices();
    auto landmarks_input_data = m_inputData->getMCurrentFrame().getMLandmarks();
    int n = (int) landmark_indices_bfm.size();
    std::cout << "Adding Residual Blocks for Sparse Optimization" << std::endl;
    auto *shape = &m_baselFaceModel->getShapeParams();
    auto *expression = &m_baselFaceModel->getExpressionParams();
    for (int i = 18; i < n; ++i) {
        if(landmarks_input_data[i].x() == -1) continue;
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimizationCost, 1, 199, 100>(
                        new SparseOptimizationCost(m_baselFaceModel, landmarks_input_data[i], landmark_indices_bfm[i])
                ),
                nullptr,
                shape->data(),
                expression->data()
                );
    }
    std::cout << "End of Adding Residual Blocks for Sparse Optimization" << std::endl;
    std::vector<double> identity_std_dev(199);
    std::vector<double> albedo_std_dev(199);
    std::vector<double> expression_std_dev(100);
    for (int i = 0; i < 199; ++i) {
        identity_std_dev[i] = std::sqrt(m_baselFaceModel->getShapePcaVariance()[i]);
        albedo_std_dev[i] = std::sqrt(m_baselFaceModel->getColorPcaVariance()[i]);
    }
    for (int i = 0; i < 100; ++i) {
        expression_std_dev[i] = std::sqrt(m_baselFaceModel->getExpressionPcaVariance()[i]);
    }

    std::cout << "Adding Residual Blocks for Regularization" << std::endl;

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ShapeRegularizerCost, 199, 199>(
            new ShapeRegularizerCost(SHAPE_REG_WEIGHT_SPARSE, m_baselFaceModel->getShapePcaVariance())
    ), nullptr, m_baselFaceModel->getShapeParams().data());

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ExpressionRegularizerCost, 100, 100>(
            new ExpressionRegularizerCost(EXPRESSION_REG_WEIGHT_SPARSE, m_baselFaceModel->getExpressionPcaVariance())
    ), nullptr, m_baselFaceModel->getExpressionParams().data());

    std::cout << "End of Adding Residual Blocks for Regularization" << std::endl;

    ceres::Solver::Summary summary;
    std::cout << "Sparse Optimization initiated." << std::endl;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Sparse Optimization finished." << std::endl;
}

void Optimizer::optimizeDenseTerms() {
    ceres::Problem problem;

    auto vertices = m_baselFaceModel->getVerticesWithoutTransformation();
    auto transformedVertices = m_baselFaceModel->transformVertices(vertices);
    int n = (int) vertices.size();
    auto correspondingPoints = m_inputData->getAllCorrespondences(transformedVertices);
    auto correspondingColors = m_inputData->getCorrespondingColors(transformedVertices);

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    int numberOfSamples = 300; //TODO: Pragma
    int maxIt = 30;             //TODO: Pragma
    int iterationCounter = 0;
    
    // Illumination
    Eigen::Matrix<double, 9, 3> shCoefficients = Illumination::loadSHCoefficients("../../../Data/face_39738.rps");

    auto* loss_function = new ceres::CauchyLoss(1.0);

    //TODO: watch out with the references! Not sure that works
    auto& expressionParams = m_baselFaceModel->getExpressionParams();
    auto& shapeParams = m_baselFaceModel->getShapeParams();
    auto& colorParams = m_baselFaceModel->getColorParams();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    m_baselFaceModel->updateNormals();

    while(iterationCounter < maxIt){
        std::cout << "Iteration: " << iterationCounter << std::endl;
        int outliers = 0;
        std::shuffle(indices.begin(), indices.end(), g);
        for (int i = 0; i < numberOfSamples; i+=1) {
            int idx = indices[i];
            Vector3d targetPoint = correspondingPoints[idx];
            auto distance = abs(transformedVertices[idx].z() - targetPoint.z());
            if(distance > OUTLIER_THRESHOLD || distance < 0.0) {
                outliers++;
                continue;
            }
            Vector3d correspondingColor = Vector3d(correspondingColors[idx].x() / 255.0, correspondingColors[idx].y() / 255.0, correspondingColors[idx].z() / 255.0);
            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<DenseOptimizationCost, 1, 199, 100>(
                            new DenseOptimizationCost(m_baselFaceModel, targetPoint, idx)
                    ),
                    loss_function,
                    shapeParams.data(),
                    expressionParams.data()
            );
            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ColorOptimizationCost, 1, 199>(
                            new ColorOptimizationCost(m_baselFaceModel, correspondingColor, shCoefficients, idx)
                    ),
                    loss_function,
                    colorParams.data()
            );
        }

        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ShapeRegularizerCost, 199, 199>(
                new ShapeRegularizerCost(SHAPE_REG_WEIGHT_DENSE, m_baselFaceModel->getShapePcaVariance())
        ), loss_function, shapeParams.data());

        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ExpressionRegularizerCost, 100, 100>(
                new ExpressionRegularizerCost(EXPRESSION_REG_WEIGHT_DENSE, m_baselFaceModel->getExpressionPcaVariance())
        ), loss_function, expressionParams.data());

        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ColorRegularizerCost, 199, 199>(
                new ColorRegularizerCost(COLOR_REG_WEIGHT_DENSE, m_baselFaceModel->getColorPcaVariance())
        ), loss_function, colorParams.data());

        ceres::Solver::Summary summary;
        std::cout << "Dense Optimization initiated." << std::endl;
        options.max_num_iterations = 15;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << std::endl;
        std::cout << "Outliers: " << outliers << std::endl;
        if(summary.termination_type == ceres::CONVERGENCE){
            break;
        }
        iterationCounter++;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time Dense Optimization (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
    std::cout << "Dense Optimization finished." << std::endl;
}

void Optimizer::configureSolver() {
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.use_nonmonotonic_steps = false; //TODO: Maybe das hier löschen
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false; //TODO: Change back to true
    options.max_num_iterations = 50;
    options.num_threads = 20;
}
