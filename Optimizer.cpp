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
    for (int i = 0; i < n; ++i) {
        if(landmarks_input_data[i].x() == -1) continue;
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimizationCost, 3, 199, 100>(
                        new SparseOptimizationCost(m_baselFaceModel, landmarks_input_data[i], landmark_indices_bfm[i])
                ),
                nullptr,
                m_baselFaceModel->getShapeParams().data(), //TODO: könnte getter hier ein problem werden?!?
                m_baselFaceModel->getExpressionParams().data()
                );
    }

    ceres::Solver::Summary summary;
    std::cout << "Sparse Optimization initiated." << std::endl;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Sparse Optimization finished." << std::endl;
}

void Optimizer::optimizeDenseGeometryTerm() {

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
    options.max_num_iterations = 1000;
    options.num_threads = 24;
}
