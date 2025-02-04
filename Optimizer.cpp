#include "Optimizer.h"

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
    for (int i = 18; i < n; ++i) {
        if(landmarks_input_data[i].x() == -1) continue;
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimizationCost, 3, 199, 100>(
                        new SparseOptimizationCost(m_baselFaceModel, landmarks_input_data[i], landmark_indices_bfm[i])
                ),
                nullptr,
                m_baselFaceModel->getShapeParams().data(),
                m_baselFaceModel->getExpressionParams().data()
                );
    }

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

    /*problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GeometryRegularizationCost, 2, 199, 100>(
                    new GeometryRegularizationCost(identity_std_dev, expression_std_dev)),
            nullptr,
            m_baselFaceModel->getShapeParams().data(),
            m_baselFaceModel->getExpressionParams().data()
    );*/
    ceres::CostFunction* shapeCost = new ceres::AutoDiffCostFunction<ShapeRegularizerCost, 199, 199>(
            new ShapeRegularizerCost()
    );
    problem.AddResidualBlock(shapeCost, nullptr, m_baselFaceModel->getShapeParams().data());

    ceres::CostFunction* expressionCost = new ceres::AutoDiffCostFunction<ExpressionRegularizerCost, 100, 100>(
            new ExpressionRegularizerCost()
    );

    problem.AddResidualBlock(expressionCost, nullptr, m_baselFaceModel->getExpressionParams().data());

    ceres::Solver::Summary summary;
    std::cout << "Sparse Optimization initiated." << std::endl;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Sparse Optimization finished." << std::endl;
}

void Optimizer::optimizeDenseGeometryTerm() {
    ceres::Problem problem;

    auto vertices = m_baselFaceModel->getVerticesWithoutTransformation();
    auto transformedVertices = m_baselFaceModel->transformVertices(vertices);
    int n = (int) vertices.size();
    for (int i = 0; i < n; i+=100) {
        Vector3d targetPoint = m_inputData->getCorrespondingPoint(transformedVertices[i]);
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<DenseOptimizationCost, 3, 199, 100>(
                        new DenseOptimizationCost(m_baselFaceModel, targetPoint, i)
                ),
                nullptr,
                m_baselFaceModel->getShapeParams().data(),
                m_baselFaceModel->getExpressionParams().data()
        );
    }
    ceres::CostFunction* shapeCost = new ceres::AutoDiffCostFunction<ShapeRegularizerCost, 199, 199>(
            new ShapeRegularizerCost()
    );
    problem.AddResidualBlock(shapeCost, nullptr, m_baselFaceModel->getShapeParams().data());

    ceres::CostFunction* expressionCost = new ceres::AutoDiffCostFunction<ExpressionRegularizerCost, 100, 100>(
            new ExpressionRegularizerCost()
    );
    problem.AddResidualBlock(expressionCost, nullptr, m_baselFaceModel->getExpressionParams().data());

    ceres::Solver::Summary summary;
    std::cout << "Dense Optimization initiated." << std::endl;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Dense Optimization finished." << std::endl;
}

void Optimizer::optimizeDenseColorTerm() {

}

void Optimizer::optimize() {
    optimizeSparseTerms();
    optimizeDenseGeometryTerm();
    optimizeDenseColorTerm();
}

void Optimizer::configureSolver() {
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.use_nonmonotonic_steps = false; //TODO: Maybe das hier löschen
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 3;
    options.num_threads = 24;
}
