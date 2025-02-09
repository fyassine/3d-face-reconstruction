#include "FaceReconstructor.h"

void FaceReconstructor::reconstructFace(BaselFaceModel *baselFaceModel, InputData *inputData, const std::string& path) {
    auto frames = inputData->getMFrames();
    for (int i = 0; i < frames.size() - 1; ++i) {
        std::string frameInputPath = path + "VideoFrames/" + std::to_string(i) + ".png";
        std::string frameOutputPath = path + "ReconstructedFace_Images/" + std::to_string(i) + ".png";
        baselFaceModel->computeTransformationMatrix(inputData);
        Optimizer optimizer(baselFaceModel, inputData);
        optimizer.optimizeSparseTerms();
        optimizer.optimizeDenseGeometryTerm();
        auto verticesAfterTransformation = baselFaceModel->getVerticesWithoutTransformation();
        auto colorAfterTransformation = baselFaceModel->getColorValues();
        auto mappedColor = inputData->getCorrespondingColors(baselFaceModel->transformVertices(verticesAfterTransformation));
        Renderer::run(baselFaceModel->transformVertices(verticesAfterTransformation), colorAfterTransformation, baselFaceModel->getFaces(), inputData->getMIntrinsicMatrix(), inputData->getMExtrinsicMatrix(), frameInputPath, frameOutputPath);
        inputData->processNextFrame();
    }
    Renderer::convertPngsToMp4(path + "ReconstructedFace_Images/", "", 55);
}

void FaceReconstructor::expressionTransfer(BaselFaceModel *sourceFaceModel, BaselFaceModel *targetFaceModel, //TODO bool saveInput?! bool useColorMap?!
                                           InputData *sourceData, InputData *targetData) {
    auto sourceFrames = sourceData->getMFrames();
    auto targetFrames = sourceData->getMFrames();
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
        optimizerSource.optimizeDenseGeometryTerm();

        Optimizer optimizerTarget(targetFaceModel, targetData);
        optimizerTarget.optimizeSparseTerms();
        optimizerTarget.optimizeDenseGeometryTerm();

        //Store source model frames
        auto sourceVertices = sourceFaceModel->transformVertices(sourceFaceModel->getVerticesWithoutTransformation());
        auto sourceColorMap = sourceData->getCorrespondingColors(sourceVertices);
        auto sourceFaces = sourceFaceModel->getFaces();
        Renderer::run(sourceVertices, sourceColorMap, sourceFaces, sourceData->getMIntrinsicMatrix(), sourceData->getMExtrinsicMatrix(), sourceFramesInputPath, sourceFramesOutputPath);

        //Store target model frames
        auto targetVertices = targetFaceModel->transformVertices(targetFaceModel->getVerticesWithoutTransformation());
        auto targetColorMap = targetData->getCorrespondingColors(targetVertices);
        auto targetFaces = targetFaceModel->getFaces();
        Renderer::run(targetVertices, targetColorMap, targetFaces, targetData->getMIntrinsicMatrix(), targetData->getMExtrinsicMatrix(), targetFramesInputPath, targetFramesOutputPath);

        //Store expression transfer frames
        targetFaceModel->getExpressionParams() = sourceFaceModel->getExpressionParams();
        auto expressionTransferVertices = targetFaceModel->transformVertices(targetFaceModel->getVerticesWithoutTransformation());
        Renderer::run(expressionTransferVertices, targetColorMap, targetFaces, targetData->getMIntrinsicMatrix(), targetData->getMExtrinsicMatrix(), targetFramesInputPath, expressionFramesOutputPath);

        sourceData->processNextFrame();
        targetData->processNextFrame();
    }

    //Save final video
    Renderer::convertPngsToMp4(resultFolderPath + "Expression_Transfer_Frames/", "Expression_Transfer_Video/", n - 1);
}
