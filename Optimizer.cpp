#include "Optimizer.h"
#include "Illumination.h"
#include "ModelConverter.h"
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
    auto* loss_function = new ceres::CauchyLoss(1.0);

    for (int i = 18; i < n; ++i) {
        if(landmarks_input_data[i].x() == -1) continue;
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SparseOptimizationCost, 3, 199, 100>(
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
            new ShapeRegularizerCost(m_shapeRegWeightSparse, m_baselFaceModel->getShapePcaVariance())
    ), loss_function, m_baselFaceModel->getShapeParams().data());

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ExpressionRegularizerCost, 100, 100>(
            new ExpressionRegularizerCost(m_expressionRegWeightSparse, m_baselFaceModel->getExpressionPcaVariance())
    ), loss_function, m_baselFaceModel->getExpressionParams().data());

    std::cout << "End of Adding Residual Blocks for Regularization" << std::endl;

    ceres::Solver::Summary summary;
    std::cout << "Sparse Optimization initiated." << std::endl;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Sparse Optimization finished." << std::endl;
}

void Optimizer::optimizeDenseTerms() {

    auto vertices = m_baselFaceModel->getVerticesWithoutTransformation();
    auto transformedVertices = m_baselFaceModel->transformVertices(vertices);
    int n = (int) vertices.size();
    auto correspondingPoints = m_inputData->getAllCorrespondences(transformedVertices);
    auto correspondingColors = m_inputData->getCorrespondingColors(transformedVertices);

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    int numberOfVerticesPerSample = n;
    int maxNumberOfSamples = 5;
    int iterationCounter = 0;

    // Illumination
    //Eigen::Matrix<double, 9, 3> shCoefficients = Illumination::loadSHCoefficients("../../../Data/face_52356.rps");
    double shCoefficients[27] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    //TODO: watch out with the references! Not sure that works
    auto& expressionParams = m_baselFaceModel->getExpressionParams();
    auto& shapeParams = m_baselFaceModel->getShapeParams();
    auto& colorParams = m_baselFaceModel->getColorParams();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //maybe keep sample from the last iteration to already have some fitting vertices?
    while(iterationCounter < maxNumberOfSamples){
        ceres::Problem problem;
        m_baselFaceModel->updateNormals();

        std::cout << "Iteration: " << iterationCounter << std::endl;
        int outliers = 0;
        std::shuffle(indices.begin(), indices.end(), g);
        std::cout << "RANDOM: " << indices[0] << std::endl;
        for (int i = 0; i < numberOfVerticesPerSample; i+=1) {
            int idx = indices[i];
            Vector3d targetPoint = correspondingPoints[idx];
            auto distance = abs(transformedVertices[idx].z() - targetPoint.z());
            if(distance > OUTLIER_THRESHOLD || distance < 0.0) {
                outliers++;
                continue;
            }
            Vector3d correspondingColor = Vector3d(correspondingColors[idx].x() / 255.0, correspondingColors[idx].y() / 255.0, correspondingColors[idx].z() / 255.0);
            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<DenseOptimizationCost, 4, 199, 100>(
                            new DenseOptimizationCost(m_baselFaceModel, targetPoint, idx)
                    ),
                    nullptr,
                    shapeParams.data(),
                    expressionParams.data()
            );
            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ColorOptimizationCost, 3, 199, 27>(
                            new ColorOptimizationCost(m_baselFaceModel, correspondingColor, idx)
                    ),
                    nullptr,
                    colorParams.data(),
                    shCoefficients
            );
        }

        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ShapeRegularizerCost, 199, 199>(
                new ShapeRegularizerCost(m_shapeRegWeightDense, m_baselFaceModel->getShapePcaVariance())
        ), nullptr, shapeParams.data());

        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ExpressionRegularizerCost, 100, 100>(
                new ExpressionRegularizerCost(m_expressionRegWeightDense, m_baselFaceModel->getExpressionPcaVariance())
        ), nullptr, expressionParams.data());

        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ColorRegularizerCost, 199, 199>(
                new ColorRegularizerCost(m_colorRegWeightDense, m_baselFaceModel->getColorPcaVariance())
        ), nullptr, colorParams.data());

        ceres::Solver::Summary summary;
        std::cout << "Dense Optimization initiated." << std::endl;
        options.max_num_iterations = 3;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
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
    /*options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.use_nonmonotonic_steps = true; //TODO: Maybe das hier löschen
    options.linear_solver_type = ceres::DENSE_QR; //sparse solver für sparse verwenden?
    options.minimizer_progress_to_stdout = true; //TODO: Change back to true
    options.max_num_iterations = 50;
    options.num_threads = 20;
    options.initial_trust_region_radius = 1e-2;  // Instead of default ~1e4
    options.min_trust_region_radius = 1e-6;     // Allow finer updates
    options.max_trust_region_radius = 10.0;*/

    options.dense_linear_algebra_library_type = ceres::CUDA;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 16;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
}

void WeightSearch::runSparseWeightTrial(const std::string &bagPath,
                          double shapeWeight,
                          double expressionWeight)
{
    // (Re)extract input data and create the face model.
    BaselFaceModel baselFaceModel;
    //InputData inputData = InputDataExtractor::extractInputData(bagPath);
    //inputData.save(dataFolderPath + "input_data.json")
    InputData inputData = InputData::load("../../../Data/input_data.json");
    baselFaceModel.computeTransformationMatrix(&inputData);

    // Create an optimizer and set the sparse weights.
    Optimizer optimizer(&baselFaceModel, &inputData);
    optimizer.setWeights(shapeWeight, expressionWeight,
                         SHAPE_REG_WEIGHT_DENSE,   // keep dense weights default
                         EXPRESSION_REG_WEIGHT_DENSE,
                         COLOR_REG_WEIGHT_DENSE);
    optimizer.configureSolver();

    // Run only the sparse optimization part.
    optimizer.optimizeSparseTerms();

    // Retrieve the vertices and corresponding colors.
    auto verticesAfterTransformation = baselFaceModel.getVerticesWithoutTransformation();
    auto transformedVertices = baselFaceModel.transformVertices(verticesAfterTransformation);
    auto mappedColor = inputData.getCorrespondingColors(transformedVertices);

    // (Optional) Apply Laplacian smoothing
    // laplacianSmooth(verticesAfterTransformation, baselFaceModel.getFaces(), 5, 0.3);

    // Build the output filename
    std::string fileName = "BfmAfterSparseTermsMappedColor_SR_" +
                           std::to_string(shapeWeight) + "_ER_" +
                           std::to_string(expressionWeight) + ".ply";

    // Write the output PLY file.
    ModelConverter::convertToPly(transformedVertices, mappedColor, baselFaceModel.getFaces(), fileName);
    //ModelConverter::generateGeometricErrorModel(&baselFaceModel, &inputData);
    std::cout << "Shape: " << shapeWeight << "; Expression: " << expressionWeight << std::endl;
}

void WeightSearch::runDenseWeightTrial(const std::string &bagPath,
                         double shapeWeight,
                         double expressionWeight,
                         double colorWeight)
{
    BaselFaceModel baselFaceModel;
    InputData inputData = InputDataExtractor::extractInputData(bagPath);
    baselFaceModel.computeTransformationMatrix(&inputData);

    Optimizer optimizer(&baselFaceModel, &inputData);
    // Keep the sparse weights at their defaults while setting dense weights.
    optimizer.setWeights(SHAPE_REG_WEIGHT_SPARSE, EXPRESSION_REG_WEIGHT_SPARSE,
                         shapeWeight, expressionWeight, colorWeight);
    optimizer.configureSolver();

    optimizer.optimizeSparseTerms();
    optimizer.optimizeDenseTerms();

    auto verticesAfterTransformation = baselFaceModel.getVerticesWithoutTransformation();
    auto transformedVertices = baselFaceModel.transformVertices(verticesAfterTransformation);
    auto mappedColor = inputData.getCorrespondingColors(transformedVertices);

    std::string fileName = "BfmAfterDenseTermsMappedColor_SD_" +
                           std::to_string(shapeWeight) + "_ED_" +
                           std::to_string(expressionWeight) + "_CD_" +
                           std::to_string(colorWeight) + ".ply";

    ModelConverter::convertToPly(transformedVertices, mappedColor, baselFaceModel.getFaces(), fileName);
}

void WeightSearch::runSparseWeightTrials(const std::string &bagPath)
{
    std::vector<double> testValues = {1, 0.8, 0.7, 0.6, 0.5};//, 0.4, 0.3, 0.2};
    std::vector<double> approximator = {0.5, 0.4, 0.3, 0.2, 0.1, 0.01};
    double defaultValue = 1.0;

    // Vary the shape weight (while keeping expression weight at default).
    /*for (double shapeWeight : testValues) {
        runSparseWeightTrial(bagPath, shapeWeight, defaultValue);
    }
    // Vary the expression weight (while keeping shape weight at default).
    for (double expressionWeight : testValues) {
        runSparseWeightTrial(bagPath, defaultValue, expressionWeight);
    }*/

    for (double app : approximator) {
        runSparseWeightTrial(bagPath, 1 / app, 0.7 / app);
    }
}

void WeightSearch::runDenseWeightTrials(const std::string &bagPath)
{
    std::vector<double> testValues = {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000};
    double defaultValue = 1.0;

    // Vary the dense shape weight.
    for (double shapeWeight : testValues) {
        runDenseWeightTrial(bagPath, shapeWeight, defaultValue, defaultValue);
    }
    // Vary the dense expression weight.
    for (double expressionWeight : testValues) {
        runDenseWeightTrial(bagPath, defaultValue, expressionWeight, defaultValue);
    }
    // Vary the dense color weight.
    for (double colorWeight : testValues) {
        runDenseWeightTrial(bagPath, defaultValue, defaultValue, colorWeight);
    }
}
