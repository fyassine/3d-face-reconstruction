#ifndef FACE_RECONSTRUCTION_RENDERING_H
#define FACE_RECONSTRUCTION_RENDERING_H

#include <dlib/opencv.h>
//#include "opencv2/imgcodecs.hpp"
#include "BFMParameters.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "FreeImage.h"

GLuint loadTexture(const char* filename) {
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

void renderQuad(GLuint texture) {
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

void renderTriangle(int verticesSize,
                    const std::vector<int>& indices,
                    unsigned int VAO){
    glBindVertexArray(VAO);
    if (indices.empty()) {
        glDrawArrays(GL_TRIANGLES, 0, verticesSize / 3);
    } else {
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    }
}

GLFWwindow* setupRendering(int width, int height){
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
    }
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Texture", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);
    glewInit();
    return window;
}

std::vector<float> setupVertexData(const std::vector<float>& vertices,
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

unsigned int setupBuffers(const std::vector<int>& indices,
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

unsigned int setupShaders(){
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;
        out vec3 vertexColor;
        void main() {
            gl_Position = vec4(aPos, 1.0);
            vertexColor = aColor;
        }
    )";

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

unsigned int setupBackgroundShaders(){
    const char* bgVertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    })";

    const char* bgFragmentShaderSource = R"(
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D texture1;
    void main() {
        FragColor = texture(texture1, TexCoord);
    })";


    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &bgVertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &bgFragmentShaderSource, NULL);
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


void renderLoop(GLuint texture,
                GLFWwindow* window,
                const std::vector<float>& vertices,
                const std::vector<int>& indices,
                unsigned int VAO){
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        //glUseProgram(setupBackgroundShaders());
        glUseProgram(0);
        renderQuad(texture);

        glUseProgram(setupShaders());
        renderTriangle(vertices.size() / 3, indices, VAO);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void cleanUp(GLuint texture, GLFWwindow* window){
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
}

static void renderFaceOnTopOfImage(int width, int height,
                                    const std::vector<float>& vertices,
                                    const std::vector<int>& indices,
                                    const std::vector<int>& colors,
                                    const char* backgroundImagePath) {
    GLFWwindow* window = setupRendering(width, height); //just take width and height of background image?! -> create background struct with texture, width and height?
    std::vector<float> vertexData = setupVertexData(vertices, colors);
    GLuint texture = loadTexture(backgroundImagePath);
    auto VAO = setupBuffers(indices, vertexData);
    //setupShaders();
    renderLoop(texture, window, vertices, indices, VAO);
    cleanUp(texture, window);
}

static BfmProperties getProperties(const std::string& path){
    BfmProperties properties;
    initializeBFM(path, properties);
    return properties;
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
    outFile << "element vertex " << 68 << std::endl;
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

    for (int i = 0; i < 68; ++i) {
        auto x = properties.landmarks[i].x();
        auto y = properties.landmarks[i].y();
        auto z = properties.landmarks[i].z();

        auto r = 255;
        auto g = 0;
        auto b = 0;

        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    outFile.close();
}

#endif //FACE_RECONSTRUCTION_RENDERING_H
