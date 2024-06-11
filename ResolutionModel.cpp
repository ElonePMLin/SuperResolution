#include "ResolutionModel.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>

void ResolutionModel::load()
{
    this->body = torch::jit::load(this->MODEL_DIR, c10::kCPU);
    return ;
}

ResolutionModel::ResolutionModel(std::string dir)
{
    this->MODEL_DIR = dir;
    this->load();
}

ResolutionModel::~ResolutionModel() {};

std::string ResolutionModel::_MODEL_DIR()
{
    return this->MODEL_DIR;
}

at::Tensor ResolutionModel::forward(at::Tensor input)
{
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    at::Tensor output = this->body.forward(inputs).toTensor();

    return output;
}

at::Tensor ResolutionModel::forward(std::vector<torch::jit::IValue> inputs)
{
    at::Tensor output = this->body.forward(inputs).toTensor();
    return output;
}

// // copy from internet
// // 模型数据转换函数
// cv::Mat tensor2Mat(at::Tensor &t)
// {
//     at::Tensor tmp = t.clone();
//     tmp = tmp.squeeze(0).detach().permute({ 1, 2, 0 });
//     tmp = tmp.mul(255).clamp(0, 255).to(torch::kU8);
//     tmp = tmp.to(torch::kCPU);
//     int h_dst = tmp.size(0);
//     int w_dst = tmp.size(1);

//     cv::Mat mat(h_dst, w_dst, CV_8UC3);
//     std::memcpy((void*)mat.data, tmp.data_ptr(), sizeof(torch::kU8) * tmp.numel());
//     return mat.clone();
// }
