#ifndef FACE_RECONSTRUCTION_RENDERING_H
#define FACE_RECONSTRUCTION_RENDERING_H

#include <dlib/opencv.h>
#include "BFMParameters.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "FreeImage.h"

static GLuint loadTexture(const char* filename) {
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename, 0);
    FIBITMAP* image = FreeImage_Load(format, filename);
    if (!image) {
        std::cerr << "Oh no! This image can not be loaded: " << filename << std::endl;
        return 0;
    }

    FIBITMAP* image32bit = FreeImage_ConvertTo32Bits(image);
    FreeImage_Unload(image);


    int width = FreeImage_GetWidth(image32bit);
    int height = FreeImage_GetHeight(image32bit);
    void* data = FreeImage_GetBits(image32bit);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    FreeImage_Unload(image32bit);

    return texture;
}

static GLuint loadTextureUpsideDown(const char* filename) {
    // Load the image
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename, 0);
    FIBITMAP* image = FreeImage_Load(format, filename);
    if (!image) {
        std::cerr << "Oh no! This image cannot be loaded: " << filename << std::endl;
        return 0;
    }

    // Convert to 32-bit format (RGBA)
    FIBITMAP* image32bit = FreeImage_ConvertTo32Bits(image);
    FreeImage_Unload(image);

    // Get width, height, and image data
    int width = FreeImage_GetWidth(image32bit);
    int height = FreeImage_GetHeight(image32bit);
    void* data = FreeImage_GetBits(image32bit);

    // Flip the image vertically
    int pitch = FreeImage_GetPitch(image32bit);
    unsigned char* rowData = new unsigned char[pitch];
    for (int y = 0; y < height / 2; ++y) {
        unsigned char* topRow = (unsigned char*)data + y * pitch;
        unsigned char* bottomRow = (unsigned char*)data + (height - y - 1) * pitch;

        // Swap rows
        memcpy(rowData, topRow, pitch);
        memcpy(topRow, bottomRow, pitch);
        memcpy(bottomRow, rowData, pitch);
    }

    delete[] rowData;

    // Generate and bind the OpenGL texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Upload the flipped image data to OpenGL
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Clean up
    FreeImage_Unload(image32bit);

    return texture;
}

static void saveFramebufferToFile(const char* filename, int width, int height) {
    // Create a buffer to store the pixel data (RGBA format)
    std::vector<unsigned char> pixels(width * height * 4); // RGBA format

    // Read pixels from the framebuffer
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // Create a FreeImage bitmap from the pixel data
    FIBITMAP* image = FreeImage_Allocate(width, height, 32); // 32-bit, RGBA format
    if (!image) {
        std::cerr << "Failed to allocate FreeImage bitmap." << std::endl;
        return;
    }

    // Flip the image vertically (since OpenGL's origin is at the bottom-left)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            //unsigned char* pixel = FreeImage_GetScanLine(image, height - y - 1) + x * 4;
            //pixel[FI_RGBA_RED] = pixels[(y * width + x) * 4 + 0];   // Red
            //pixel[FI_RGBA_GREEN] = pixels[(y * width + x) * 4 + 1]; // Green
            //pixel[FI_RGBA_BLUE] = pixels[(y * width + x) * 4 + 2];  // Blue
            //pixel[FI_RGBA_ALPHA] = pixels[(y * width + x) * 4 + 3]; // Alpha
        }
    }

    // Save the image as PNG or JPG
    if (!FreeImage_Save(FIF_PNG, image, filename)) {
        std::cerr << "Failed to save image to file." << std::endl;
    }

    // Unload the image
    FreeImage_Unload(image);
}

static void renderQuad(GLuint texture) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
}

static void renderTriangle(int verticesSize,
                    const std::vector<int>& indices,
                    unsigned int VAO){
    glBindVertexArray(VAO);
    if (indices.empty()) {
        glDrawArrays(GL_TRIANGLES, 0, verticesSize / 3);
    } else {
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    }
}

static GLFWwindow* setupRendering(int width, int height){
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }
    GLFWwindow* window = glfwCreateWindow(width, height, "Output", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    //glewExperimental = GL_TRUE; // Ensure experimental extensions are enabled
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return nullptr;
    }

    //glFrontFace(GL_CCW);
    //glewInit();
    return window;
}

static std::vector<float> setupVertexData(const std::vector<float>& vertices,
                                   const std::vector<int>& colors){
    std::vector<float> vertexData;
    for (size_t i = 0; i < vertices.size() / 3; ++i) {
        vertexData.push_back(vertices[i * 3 + 0]);                  // x
        vertexData.push_back(vertices[i * 3 + 1]);                  // y
        vertexData.push_back(vertices[i * 3 + 2]);                  // z
        vertexData.push_back((float) colors[i * 3 + 0] / 255.0f);   // r
        vertexData.push_back((float) colors[i * 3 + 1] / 255.0f);   // g
        vertexData.push_back((float) colors[i * 3 + 2] / 255.0f);   // b
    }
    return vertexData;
}

static unsigned int setupBuffers(const std::vector<int>& indices,
                  const std::vector<float>& vertexData){
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

    if (!indices.empty()) {
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    }

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    return VAO;
}

static unsigned int setupShaders(){
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;

        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 projection;

        out vec3 vertexColor;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            vertexColor = aColor;
        }
    )";
//TODO Work on model matrix

    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 vertexColor;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(vertexColor, 1.0);
        }
    )";

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    //glUseProgram(shaderProgram);
    return shaderProgram;
}

static Eigen::Matrix4f projectionFromIntrinsics(const Eigen::Matrix3f& intrinsics, float near_plane, float far_plane, int width, int height) {
    float fx = intrinsics(0, 0);
    float fy = intrinsics(1, 1);
    float cx = intrinsics(0, 2);
    float cy = intrinsics(1, 2);

    // Compute frustum extents
    float l = -(cx / fx) * near_plane;
    float r = ((width - cx) / fx) * near_plane;
    float b = -(cy / fy) * near_plane;
    float t = ((height - cy) / fy) * near_plane;

    Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
    projection(0, 0) = 2 * near_plane / (r - l);
    projection(1, 1) = 2 * near_plane / (t - b);
    projection(2, 0) = (r + l) / (r - l);
    projection(2, 1) = (t + b) / (t - b);
    projection(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
    projection(2, 3) = 1.01f;
    projection(3, 2) = -(2 * far_plane * near_plane) / (far_plane - near_plane);
    projection(3, 3) = 0.0f;

    return projection;
}
/*static Eigen::Matrix4f projectionFromIntrinsics(const Eigen::Matrix3f& intrinsics, float near_plane, float far_plane, int width, int height) {
    float fx = intrinsics(0, 0);
    float fy = intrinsics(1, 1);
    float cx = intrinsics(0, 2);
    float cy = intrinsics(1, 2);

    float l = -cx * near_plane / fx;
    float r = (width - cx) * near_plane / fx;
    float b = -cy * near_plane / fy;
    float t = (height - cy) * near_plane / fy;

    Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
    projection(0, 0) = 2 * near_plane / (r - l);
    projection(1, 1) = 2 * near_plane / (t - b);  // Fixed sign issue
    projection(2, 0) = (r + l) / (r - l);
    projection(2, 1) = (t + b) / (t - b);
    projection(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
    projection(2, 3) = -(2 * far_plane * near_plane) / (far_plane - near_plane); // Fixed incorrect placement
    projection(3, 2) = -1;  // Fixed incorrect placement
    projection(3, 3) = 0.0f;

    return projection;
}*/

static GLfloat* eigenToOpenGL(const Eigen::Matrix4f& mat) {
    GLfloat* glMat = new GLfloat[16];
    for (int i = 0; i < 16; ++i) {
        glMat[i] = mat.data()[i];
    }
    return glMat;
}

static void renderLoop(GLuint texture,
                GLFWwindow* window,
                const std::vector<float>& vertices,
                const std::vector<int>& indices,
                unsigned int VAO, const InputImage& inputImage, const Eigen::Matrix4f& modelTransform){
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        glUseProgram(0);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        renderQuad(texture);

        GLuint shaderProgram = setupShaders();
        glUseProgram(shaderProgram);
        Eigen::Matrix4f projection = projectionFromIntrinsics(inputImage.intrinsics, 0.001f, 100.0f, 1280, 720);
        //Eigen::Matrix4f view = inputImage.extrinsics.inverse(); //inverse?!

        GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");


        Eigen::Matrix3f rotation = inputImage.extrinsics.block<3,3>(0,0).transpose();  // Invert rotation
        Eigen::Vector3f translation = -rotation * inputImage.extrinsics.block<3,1>(0,3);  // Adjust translation
        Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
        //view(1, 1) = -view(1, 1);
        //view(0, 0) = -view(0, 0);
        //view(2, 2) = -view(2, 2);
        //view.block<3,3>(0,0) = rotation;
        //view.block<3,1>(0,3) = translation;

        glUniformMatrix4fv(projectionLoc, 1, GL_TRUE, eigenToOpenGL(projection));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, eigenToOpenGL(view));
        auto modelTransformMatrix = modelTransform;
        //modelTransformMatrix(2, 3) = -modelTransform(2, 3); // Move model in front of the camera
        //modelTransformMatrix(1, 1) *= -1; // Flip upside down
        //modelTransformMatrix(2, 3) = 3;//-modelTransform(2, 3);
        std::cout << "Projection Matrix:\n" << projection << std::endl;
        std::cout << "View Matrix (Extrinsics Inverted):\n" << view << std::endl;
        std::cout << "Model Matrix:\n" << modelTransformMatrix << std::endl;
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, eigenToOpenGL(modelTransformMatrix));
        renderTriangle(vertices.size() / 3, indices, VAO);
        saveFramebufferToFile((resultFolderPath + "rendering.png").c_str(), 1280, 720);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

static void cleanUp(GLuint texture, GLFWwindow* window){
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
}

static void renderFaceOnTopOfImage(int width, int height,
                                    const std::vector<float>& vertices,
                                    const std::vector<int>& indices,
                                    const std::vector<int>& colors,
                                    const char* backgroundImagePath, const InputImage& inputImage, const Eigen::Matrix4f& modelTransform) {
    GLFWwindow* window = setupRendering(width, height);
    std::vector<float> vertexData = setupVertexData(vertices, colors);
    GLuint texture = loadTexture(backgroundImagePath);
    auto VAO = setupBuffers(indices, vertexData);
    renderLoop(texture, window, vertices, indices, VAO, inputImage, modelTransform);
    cleanUp(texture, window);
}

static BfmProperties getProperties(const std::string& path, const InputImage& inputImage){
    BfmProperties properties;
    initializeBFM(path, properties, inputImage);
    return properties;
}

static void convertParametersToPlyWithoutProcrustes(const BfmProperties& properties, const std::string& resultPath){

    std::ofstream outFile(resultPath);
    //Header
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << properties.numberOfVertices << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << properties.numberOfTriangles << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;
    //Vertices
    auto vertices = getVerticesWithoutProcrustes(properties);
    auto colorValues = getColorValues(properties);
    for (int i = 0; i < properties.numberOfVertices; i++) {
        //Position
        auto x = vertices[i].x();
        auto y = vertices[i].y();
        auto z = vertices[i].z();
        //Color
        auto r = colorValues[i].x();
        auto g = colorValues[i].y();
        auto b = colorValues[i].z();
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }

    //Faces
    for (int i = 0; i < properties.numberOfTriangles * 3; i+=3) {
        outFile << "3 " << properties.triangles[i] << " " << properties.triangles[i + 1] << " " << properties.triangles[i + 2] << std::endl;
    }
    outFile.close();
}

static void convertParametersToPly(const BfmProperties& properties, const std::string& resultPath){

    std::ofstream outFile(resultPath);
    //Header
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << properties.numberOfVertices << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << properties.numberOfTriangles << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;
    //Vertices
    auto vertices = getVertices(properties);
    auto colorValues = getColorValues(properties);
    for (int i = 0; i < properties.numberOfVertices; i++) {
        //Position
        auto x = vertices[i].x();
        auto y = vertices[i].y();
        auto z = vertices[i].z();
        //Color
        auto r = colorValues[i].x();
        auto g = colorValues[i].y();
        auto b = colorValues[i].z();
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }

    //Faces
    for (int i = 0; i < properties.numberOfTriangles * 3; i+=3) {
        outFile << "3 " << properties.triangles[i] << " " << properties.triangles[i + 1] << " " << properties.triangles[i + 2] << std::endl;
    }
    outFile.close();
}

static void convertLandmarksToPly(const BfmProperties& properties, const std::string& resultPath){

    std::ofstream outFile(resultPath);
    //Header
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << properties.landmarks.size() << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << 0 << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;
    //Vertices

    for (int i = 0; i < properties.landmarks.size(); ++i) {
        Eigen::Vector4f transformedLandmark(properties.landmarks[i].x(), properties.landmarks[i].y(), properties.landmarks[i].z(), 1.0f);
        auto currentLandmark = properties.transformation * transformedLandmark;

        auto x = currentLandmark.x();
        auto y = currentLandmark.y();
        auto z = currentLandmark.z();

        auto r = 255;
        auto g = 0;
        auto b = 0;

        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    outFile.close();
}

static void getPointCloud(const std::vector<Eigen::Vector2f>& vertices, const std::vector<float>& depth, const std::vector<Eigen::Vector3i>& colorValues, const std::string& resultPath,
                          const Matrix3f& depthIntrinsics, const Matrix4f& extrinsics){

    std::ofstream outFile(resultPath);
    //Header
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << vertices.size() << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << 0 << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;
    //Vertices
    for (int i = 0; i < vertices.size(); i++) {
        //Position
        Eigen::Vector3f vertex3D = convert2Dto3D(vertices[i], depth[i], depthIntrinsics, extrinsics);

        auto x = (float) vertex3D.x();
        auto y = (float) vertex3D.y();
        auto z = (float) vertex3D.z();
        //Color
        auto r = colorValues[i].x();
        auto g = colorValues[i].y();
        auto b = colorValues[i].z();
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    outFile.close();
}

static void convertVerticesTest(const std::vector<Eigen::Vector3f>& vertices, const std::string& resultPath){

    std::ofstream outFile(resultPath);
    //Header
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << vertices.size() << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << 0 << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;
    //Vertices

    for (int i = 0; i < vertices.size(); ++i) {
        auto x = vertices[i].x();
        auto y = vertices[i].y();
        auto z = vertices[i].z();

        auto r = 255;
        auto g = 0;
        auto b = 0;

        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    outFile.close();
}

#endif //FACE_RECONSTRUCTION_RENDERING_H
