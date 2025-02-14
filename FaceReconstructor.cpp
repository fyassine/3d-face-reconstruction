#include "FaceReconstructor.h"

void FaceReconstructor::reconstructFace(BaselFaceModel *baselFaceModel, InputData *inputData) {
    auto sourceFrames = inputData->getMFrames();
    int n = sourceFrames.size();
    std::string resultFolderPath = "../../../Result/";
    baselFaceModel->computeTransformationMatrix(inputData);

    for (int i = 0; i < n - 1; ++i) {
        std::string sourceFramesInputPath = resultFolderPath + "Source_Frames/" + std::to_string(i) + ".png";
        std::string sourceFramesOutputPath = resultFolderPath + "Source_Frames_Reconstructed/" + std::to_string(i) + ".png";
        std::string sourceFramesTextureOutputPath = resultFolderPath + "Source_Frames_Reconstructed_Texture/" + std::to_string(i) + ".png";

        //Store Backprojection
        ModelConverter::convertImageToPly(inputData->getMCurrentFrame().getMDepthData(), inputData->getMCurrentFrame().getMRgbData(), "Source_Backprojections/" + std::to_string(i) + ".ply", inputData->getMIntrinsicMatrix(), inputData->getMExtrinsicMatrix());


        //Store input frames
        Renderer::convertColorToPng(inputData->getMCurrentFrame().getMRgbData(), sourceFramesInputPath);

        //Store Results after Procrustes
        ModelConverter::convertBFMToPly(baselFaceModel, "Source_Models_Procrustes/" + std::to_string(i) + ".ply");
        ModelConverter::generateGeometricErrorModel(baselFaceModel, inputData, "Source_GeometricError_Procrustes/" + std::to_string(i) + ".ply");

        //Optimization
        Optimizer optimizerSource(baselFaceModel, inputData);
        optimizerSource.optimizeSparseTerms();

        //Store Results after Sparse
        ModelConverter::convertBFMToPly(baselFaceModel, "Source_Models_Sparse/" + std::to_string(i) + ".ply");
        ModelConverter::generateGeometricErrorModel(baselFaceModel, inputData, "Source_GeometricError_Sparse/" + std::to_string(i) + ".ply");

        optimizerSource.optimizeDenseTerms();

        //Store Results after Dense
        ModelConverter::convertBFMToPly(baselFaceModel, "Source_Models_Dense/" + std::to_string(i) + ".ply");
        ModelConverter::generateGeometricErrorModel(baselFaceModel, inputData, "Source_GeometricError_Dense/" + std::to_string(i) + ".ply");

        auto colorMap = inputData->getCorrespondingColors(baselFaceModel->transformVertices(baselFaceModel->getVerticesWithoutTransformation()));
        ModelConverter::convertToPly(baselFaceModel->transformVertices(baselFaceModel->getVerticesWithoutTransformation()), colorMap, baselFaceModel->getFaces(), "Source_Models_TextureMap/" + std::to_string(i) + ".ply");

        //Store source model frames
        auto sourceVertices = baselFaceModel->transformVertices(baselFaceModel->getVerticesWithoutTransformation());
        auto sourceColorMap = inputData->getCorrespondingColors(sourceVertices);
        auto sourceColor = baselFaceModel->getColorValues();
        auto sourceFaces = baselFaceModel->getFaces();
        Renderer::run(sourceVertices, sourceColor, sourceFaces, inputData->getMIntrinsicMatrix(), inputData->getMExtrinsicMatrix(), sourceFramesInputPath, sourceFramesOutputPath);
        Renderer::run(sourceVertices, colorMap, sourceFaces, inputData->getMIntrinsicMatrix(), inputData->getMExtrinsicMatrix(), sourceFramesInputPath, sourceFramesOutputPath);

        inputData->processNextFrame();
    }

    //Save final video
    //Renderer::convertPngsToMp4(resultFolderPath + "Expression_Frames_Reconstructed/", "Expression_Transfer_Video/", n - 1);
}

void FaceReconstructor::expressionTransfer(BaselFaceModel *sourceFaceModel, BaselFaceModel *targetFaceModel,
                                           InputData *sourceData, InputData *targetData) {
    auto sourceFrames = sourceData->getMFrames();
    auto targetFrames = targetData->getMFrames();
    int n = sourceFrames.size() > targetFrames.size() ? (int) targetFrames.size() : (int) sourceFrames.size();
    std::string resultFolderPath = "../../../Result/";
    sourceFaceModel->computeTransformationMatrix(sourceData);
    targetFaceModel->computeTransformationMatrix(targetData);

    for (int i = 0; i < n - 1; ++i) {
        std::string sourceFramesInputPath = resultFolderPath + "Source_Frames/" + std::to_string(i) + ".png";
        std::string targetFramesInputPath = resultFolderPath + "Target_Frames/" + std::to_string(i) + ".png";
        std::string sourceFramesOutputPath = resultFolderPath + "Source_Frames_Reconstructed/" + std::to_string(i) + ".png";
        std::string targetFramesOutputPath = resultFolderPath + "Target_Frames_Reconstructed/" + std::to_string(i) + ".png";
        std::string expressionFramesOutputPath = resultFolderPath + "Expression_Frames_Reconstructed/" + std::to_string(i) + ".png";

        //Store input frames
        Renderer::convertColorToPng(sourceData->getMCurrentFrame().getMRgbData(), sourceFramesInputPath);
        Renderer::convertColorToPng(targetData->getMCurrentFrame().getMRgbData(), targetFramesInputPath);

        //Optimization
        Optimizer optimizerSource(sourceFaceModel, sourceData);
        optimizerSource.optimizeSparseTerms();
        optimizerSource.optimizeDenseTerms();

        Optimizer optimizerTarget(targetFaceModel, targetData);
        optimizerTarget.optimizeSparseTerms();
        optimizerTarget.optimizeDenseTerms();

        //Store source model frames
        auto sourceVertices = sourceFaceModel->transformVertices(sourceFaceModel->getVerticesWithoutTransformation());
        auto sourceColorMap = sourceData->getCorrespondingColors(sourceVertices);
        auto sourceColor = sourceFaceModel->getColorValues();
        auto sourceFaces = sourceFaceModel->getFaces();
        Renderer::run(sourceVertices, sourceColor, sourceFaces, sourceData->getMIntrinsicMatrix(), sourceData->getMExtrinsicMatrix(), sourceFramesInputPath, sourceFramesOutputPath);

        //Store target model frames
        auto targetVertices = targetFaceModel->transformVertices(targetFaceModel->getVerticesWithoutTransformation());
        auto targetColorMap = targetData->getCorrespondingColors(targetVertices);
        auto targetColor = targetFaceModel->getColorValues();
        auto targetFaces = targetFaceModel->getFaces();
        Renderer::run(targetVertices, targetColor, targetFaces, targetData->getMIntrinsicMatrix(), targetData->getMExtrinsicMatrix(), targetFramesInputPath, targetFramesOutputPath);

        //Store expression transfer frames
        targetFaceModel->getExpressionParams() = sourceFaceModel->getExpressionParams();
        auto expressionTransferVertices = targetFaceModel->transformVertices(targetFaceModel->getVerticesWithoutTransformation());
        Renderer::run(expressionTransferVertices, targetColorMap, targetFaces, targetData->getMIntrinsicMatrix(), targetData->getMExtrinsicMatrix(), targetFramesInputPath, expressionFramesOutputPath);

        sourceData->processNextFrame();
        targetData->processNextFrame();
    }

    //Save final video
    Renderer::convertPngsToMp4(resultFolderPath + "Expression_Frames_Reconstructed/", "Expression_Transfer_Video/", n - 1);
}
