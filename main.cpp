#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "PointCloud.h"

#include "FacialLandmarks.h"
#include "Optimization.h"
#include "Rendering.h"
#include "BFMParameters.h"
#include "ProcrustesAligner.h"
#include "ImageExtraction.h"
#include "BaselFaceModel.h"

#include "Renderer.h"
//#include "InputDataExtractor.h"
#include "InputData.h"
#include "Optimizer.h"
#include "ModelConverter.h"

using namespace Eigen;
using namespace std;

#define LEO_LOOKING_NORMAL "20250127_200932.bag"
#define NELI_LOOKING_SERIOUS "20250116_183206.bag"
#define LEO_CRAZY "20250201_195224.bag"
#define LEO_LONG "20250205_172132.bag"

BaselFaceModel processFace(const std::string& path, InputData* inputData){
    BaselFaceModel baselFaceModel;

    baselFaceModel.computeTransformationMatrix(inputData);

    auto verticesBeforeSparse = baselFaceModel.getVerticesWithoutTransformation();
    auto colorBeforeSparse = baselFaceModel.getColorValues();
    ModelConverter::convertToPly(verticesBeforeSparse, colorBeforeSparse, baselFaceModel.getFaces(), "BfmBeforeSparseTerms.ply");

    auto landmarksBeforeSparse = baselFaceModel.getLandmarks();
    ModelConverter::convertToPly(landmarksBeforeSparse, "LandmarksBeforeSparse.ply");

    ModelConverter::convertToPly(baselFaceModel.transformVertices(verticesBeforeSparse), colorBeforeSparse, baselFaceModel.getFaces(), "ModelAfterProcrustes.ply");
    ModelConverter::convertToPly(baselFaceModel.transformVertices(landmarksBeforeSparse), "LandmarksAfterProcrustes.ply");

    auto landmarksOfInputData = inputData->getMCurrentFrame().getMLandmarks();
    ModelConverter::convertToPly(landmarksOfInputData, "LandmarksOfInputImage.ply");

    ModelConverter::convertImageToPly(inputData->getMCurrentFrame().getMDepthData(), inputData->getMCurrentFrame().getMRgbData(), "BackprojectedImage.ply", inputData->getMIntrinsicMatrix(), inputData->getMExtrinsicMatrix());

    Optimizer optimizer(&baselFaceModel, inputData);
    optimizer.optimizeSparseTerms();

    auto verticesAfterTransformation = baselFaceModel.getVerticesWithoutTransformation();
    auto colorAfterTransformation = baselFaceModel.getColorValues();
    auto mappedColor = inputData->getCorrespondingColors(baselFaceModel.transformVertices(verticesAfterTransformation));
    ModelConverter::convertToPly(verticesAfterTransformation, colorAfterTransformation, baselFaceModel.getFaces(), "BfmAfterSparseTerms.ply");
    ModelConverter::convertToPly(baselFaceModel.transformVertices(verticesAfterTransformation), mappedColor, baselFaceModel.getFaces(), "BfmAfterSparseTermsMappedColor.ply");
    //Renderer::run(baselFaceModel.transformVertices(verticesAfterTransformation), mappedColor, baselFaceModel.getFaces(), inputData->getMIntrinsicMatrix(), inputData->getMExtrinsicMatrix());

    auto inputVertices = inputData->getAllCorrespondences(baselFaceModel.transformVertices(verticesAfterTransformation));
    ModelConverter::convertToPly(inputVertices, colorAfterTransformation, baselFaceModel.getFaces(), "CorrespondencesBfm.ply");

    auto landmarksAfterSparse = baselFaceModel.getLandmarks();
    ModelConverter::convertToPly(landmarksAfterSparse, "LandmarksAfterSparse.ply");

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
    ModelConverter::convertToPly(baselFaceModel.transformVertices(verticesAfterTransformation), mappedColor, baselFaceModel.getFaces(), "BfmAfterDenseTermsMappedColor.ply");
*/
    return baselFaceModel;
}

int main(){
    InputData inputLeo = InputDataExtractor::extractInputData(LEO_CRAZY);
    InputData inputNeli = InputDataExtractor::extractInputData(NELI_LOOKING_SERIOUS);

    auto sourceFace = processFace(LEO_LOOKING_NORMAL, &inputLeo);
    auto targetFace = processFace(NELI_LOOKING_SERIOUS, &inputNeli);
    auto verticesAfterTransformation = targetFace.getVerticesWithoutTransformation();
    auto mappedColor = inputNeli.getCorrespondingColors(targetFace.transformVertices(verticesAfterTransformation));

    targetFace.expressionTransfer(&sourceFace);

    verticesAfterTransformation = targetFace.getVerticesWithoutTransformation();

    auto colorAfterTransformation = targetFace.getColorValues();

    ModelConverter::convertToPly(targetFace.transformVertices(verticesAfterTransformation), colorAfterTransformation, targetFace.getFaces(), "ExpressionTransfer.ply");
    Renderer::run(targetFace.transformVertices(verticesAfterTransformation), mappedColor, targetFace.getFaces(), inputNeli.getMIntrinsicMatrix(), inputNeli.getMExtrinsicMatrix());

    //TODO: Create Renderer
}
