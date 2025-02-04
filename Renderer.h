#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

struct Face {
    unsigned int v1, v2, v3;
};

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();
    void setupCamera(const Eigen::Matrix3f& intrinsics);
    void renderModel(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Face>& faces);
    void run(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Face>& faces);

private:
    GLFWwindow* window;
    int width, height;
    Eigen::Matrix3f intrinsics;
};

