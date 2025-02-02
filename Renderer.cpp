#include "Renderer.h"

Renderer::Renderer(int width, int height) : width(width), height(height), window(nullptr) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(-1);
    }

    window = glfwCreateWindow(width, height, "Rendered Face", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        exit(-1);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black
}

Renderer::~Renderer() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Renderer::setupCamera(const Eigen::Matrix3f& intrinsics) {
    this->intrinsics = intrinsics;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float fx = intrinsics(0, 0);
    float fy = intrinsics(1, 1);
    float cx = intrinsics(0, 2);
    float cy = intrinsics(1, 2);

    float left = -cx / fx;
    float right = ((float) width - cx) / fx;
    float bottom = -(cy - (float) height) / fy;
    float top = cy / fy;
    float near = 0.1f, far = 100.0f;

    glFrustum(left, right, bottom, top, near, far);
    glMatrixMode(GL_MODELVIEW);
}

void Renderer::renderModel(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Face>& faces) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -0.0f);

    glBegin(GL_TRIANGLES);
    for (const auto& face : faces) {
        glColor3f(1.0f, 0.0f, 0.0f); // Set color to red
        glVertex3f(vertices[face.v1].x(), vertices[face.v1].y(), vertices[face.v1].z());
        glColor3f(0.0f, 1.0f, 0.0f); // Set color to green
        glVertex3f(vertices[face.v2].x(), vertices[face.v2].y(), vertices[face.v2].z());
        glColor3f(0.0f, 0.0f, 1.0f); // Set color to blue
        glVertex3f(vertices[face.v3].x(), vertices[face.v3].y(), vertices[face.v3].z());
    }
    glEnd();
}

void Renderer::run(const std::vector<Eigen::Vector3f>& vertices, const std::vector<Face>& faces) {
    while (!glfwWindowShouldClose(window)) {
        renderModel(vertices, faces);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}
