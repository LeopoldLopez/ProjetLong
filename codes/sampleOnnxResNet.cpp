#include "NvInfer.h"
#include "parserOnnxConfig.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <memory>
#include "logger.h"

namespace fs = std::filesystem;
using namespace nvinfer1;

const std::string modelPath = "data/resnet50/ResNet50.onnx";
const std::string imageFolder = "data/resnet50/";
const std::string labelsPath = "data/resnet50/class_labels.txt";
const std::string inputTensorName = "gpu_0/data_0";
const std::string outputTensorName = "gpu_0/softmax_1";
const int BATCH_SIZE = 1;
const int INPUT_H = 224;
const int INPUT_W = 224;
const int INPUT_C = 3;
const int OUTPUT_SIZE = 1000; // Nombre de classes de ResNet50

std::vector<std::string> loadLabels(const std::string& filePath) {
    std::vector<std::string> labels;
    std::ifstream file(filePath);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

std::string getRandomImage(const std::string& folder) {
    std::vector<std::string> images;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            images.push_back(entry.path().string());
        }
    }
    if (images.empty()) {
        throw std::runtime_error("No valid images found in " + folder);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, images.size() - 1);
    return images[dis(gen)];
}

void preprocessImage(const std::string& imagePath, std::vector<float>& inputTensor) {
    cv::Mat img = cv::imread(imagePath);
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
    
    // Normalization (mean and std for ResNet50)
    cv::Mat channels[3];
    cv::split(img, channels);
    float mean[] = {0.485, 0.456, 0.406};
    float std[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    cv::merge(channels, 3, img);
    
    // Convert to CHW format with batch dimension
    int index = 0;
    for (int c = 0; c < INPUT_C; ++c) {
        for (int i = 0; i < INPUT_H; ++i) {
            for (int j = 0; j < INPUT_W; ++j) {
                inputTensor[index++] = img.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
}

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

int main() {
    Logger logger;
    
    // Charger les labels
    std::vector<std::string> labels = loadLabels(labelsPath);
    if (labels.size() != OUTPUT_SIZE) {
        throw std::runtime_error("Mismatch between label count and output size");
    }
    
    // Charger l'image
    std::string imagePath = getRandomImage(imageFolder);
    std::cout << "Using image: " << imagePath << std::endl;
    std::vector<float> inputTensor(BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W);
    preprocessImage(imagePath, inputTensor);
    
    // Création du runtime TensorRT
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    std::ifstream modelFile(modelPath, std::ios::binary);
    modelFile.seekg(0, std::ios::end);
    size_t modelSize = modelFile.tellg();
    modelFile.seekg(0, std::ios::beg);
    std::vector<char> modelData(modelSize);
    modelFile.read(modelData.data(), modelSize);
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelSize));
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    
    // Allocation mémoire GPU
    void* gpuInput;
    void* gpuOutput;
    cudaMalloc(&gpuInput, inputTensor.size() * sizeof(float));
    cudaMalloc(&gpuOutput, OUTPUT_SIZE * sizeof(float));
    
    // Copier l'entrée vers le GPU
    cudaMemcpy(gpuInput, inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Exécuter l'inférence
    void* bindings[2] = {gpuInput, gpuOutput};
    context->executeV2(bindings);
    
    // Copier les résultats vers le CPU
    float outputTensor[OUTPUT_SIZE];
    cudaMemcpy(outputTensor, gpuOutput, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Trouver la classe avec la plus grande probabilité
    int maxIdx = std::max_element(outputTensor, outputTensor + OUTPUT_SIZE) - outputTensor;
    std::cout << "Predicted class: " << labels[maxIdx] << " (" << maxIdx << ")" << std::endl;
    
    // Libérer la mémoire GPU
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    
    return 0;
}

