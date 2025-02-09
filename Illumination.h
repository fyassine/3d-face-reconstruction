#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Illumination {
    
    static Eigen::Matrix<double, 9, 3> loadSHCoefficients(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        json j;
        file >> j;
        
        Eigen::Matrix<double, 9, 3> shCoeffs;
        auto coefficients = j["environmentMap"]["coefficients"];
        
        for (int i = 0; i < 9; ++i) {
            shCoeffs(i, 0) = coefficients[i][0]; // Red
            shCoeffs(i, 1) = coefficients[i][1]; // Green
            shCoeffs(i, 2) = coefficients[i][2]; // Blue
        }
        
        return shCoeffs;
    }
};
