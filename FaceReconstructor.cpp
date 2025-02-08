#include "FaceReconstructor.h"

void FaceReconstructor::reconstructFace(BaselFaceModel *baselFaceModel, InputData *inputData, const std::string& path) {
    auto frames = inputData->getMFrames();
    for (int i = 0; i < frames.size() - 1; ++i) {
        std::string frameInputPath = path + "VideoFrames/" + std::to_string(i) + ".png";
        std::string frameOutputPath = path + "ReconstructedFace_Images/" + std::to_string(i) + ".png";
        baselFaceModel->computeTransformationMatrix(inputData);
        Optimizer optimizer(baselFaceModel, inputData);
        optimizer.optimizeSparseTerms();
        auto verticesAfterTransformation = baselFaceModel->getVerticesWithoutTransformation();
        auto colorAfterTransformation = baselFaceModel->getColorValues();
        auto mappedColor = inputData->getCorrespondingColors(baselFaceModel->transformVertices(verticesAfterTransformation));
        Renderer::run(baselFaceModel->transformVertices(verticesAfterTransformation), colorAfterTransformation, baselFaceModel->getFaces(), inputData->getMIntrinsicMatrix(), inputData->getMExtrinsicMatrix(), frameInputPath, frameOutputPath);
        inputData->processNextFrame();
    }

    Renderer::convertPngsToMp4(path + "ReconstructedFace_Images/", "", 55);

    //TODO: Smth wrong with reg for dense
    /*optimizer.optimizeDenseGeometryTerm();
    verticesAfterTransformation = baselFaceModel.getVerticesWithoutTransformation();
    auto transformedVerticesDense = baselFaceModel.transformVertices(verticesAfterTransformation);
    colorAfterTransformation = baselFaceModel.getColorValues();
    ModelConverter::convertToPly(verticesAfterTransformation, colorAfterTransformation, baselFaceModel.getFaces(), "BfmAfterDenseTerms.ply");
    ModelConverter::convertToPly(transformedVerticesDense, colorAfterTransformation, baselFaceModel.getFaces(), "BfmAfterDenseTermsProcrustes.ply");
    auto landmarksAfterDense = baselFaceModel.getLandmarks();
    ModelConverter::convertToPly(landmarksAfterDense, "LandmarksAfterDense.ply");
    mappedColor = inputData.getCorrespondingColors(baselFaceModel.transformVertices(verticesAfterTransformation));
    ModelConverter::convertToPly(baselFaceModel.transformVertices(verticesAfterTransformation), mappedColor, baselFaceModel.getFaces(), "BfmAfterDenseTermsMappedColor.ply");*/
}
