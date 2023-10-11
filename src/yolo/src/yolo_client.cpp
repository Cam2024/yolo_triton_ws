
#include <iostream>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include<geometry_msgs/Twist.h>
#include <geometry_msgs/TransformStamped.h>


#include <tritonclient/grpc_client.h>
#include <tritonclient/http_client.h>

namespace ti = triton;
namespace tc = triton::client;
ros::Publisher pub;

class Param {
public:
    std::string model = "yolov5_new";
    int width = 640;
    int height = 640;
    std::string url = "localhost:8001";
    float confidence = 0.85;
    float nms = 0.45;
    bool model_info = false;
    bool verbose = false;
    int client_timeout = 0;

    Param() {}  // 默认构造函数
};

class BoundingBox {
public:
    int classID;
    float confidence;
    float x1, x2, y1, y2;
    int image_width, image_height;
    float u1, u2, v1, v2;

    BoundingBox(int cls, float conf, float x1, float x2, float y1, float y2, int img_w, int img_h)
        : classID(cls), confidence(conf), x1(x1), x2(x2), y1(y1), y2(y2), image_width(img_w), image_height(img_h) {
        u1 = x1 / image_width;
        u2 = x2 / image_width;
        v1 = y1 / image_height;
        v2 = y2 / image_height;
    }

    std::tuple<float, float, float, float> box() {
        return std::make_tuple(x1, y1, x2, y2);
    }

    float width() {
        return x2 - x1;
    }

    float height() {
        return y2 - y1;
    }

    std::tuple<float, float> center_absolute() {
        return std::make_tuple(0.5 * (x1 + x2), 0.5 * (y1 + y2));
    }

    std::tuple<float, float> center_normalized() {
        return std::make_tuple(0.5 * (u1 + u2), 0.5 * (v1 + v2));
    }

    std::tuple<float, float> size_absolute() {
        return std::make_tuple(x2 - x1, y2 - y1);
    }

    std::tuple<float, float> size_normalized() {
        return std::make_tuple(u2 - u1, v2 - v1);
    }
};

struct TritonModelInfo {
    std::string output_name_;
    std::vector<std::string> output_names_;
    std::string input_name_;
    std::string input_datatype_;
    // The shape of the input
    int input_c_;
    int input_h_;
    int input_w_;
    // The format of the input
    std::string input_format_;
    int type1_;
    int type3_;
    int max_batch_size_;

    std::vector<int64_t> shape_;

};

union TritonClient {
  TritonClient()
  {
    new (&http_client_) std::unique_ptr<tc::InferenceServerHttpClient>{};
  }
  ~TritonClient() {}

  std::unique_ptr<tc::InferenceServerHttpClient> http_client_;
  std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client_;
};



void setModel(TritonModelInfo& yoloModelInfo, const int batch_size){
    yoloModelInfo.output_names_ = std::vector<std::string>{"prob"};
    yoloModelInfo.input_name_ = "data";
    yoloModelInfo.input_datatype_ = std::string("FP32");
    // The shape of the input
    yoloModelInfo.input_c_ = 3;
    yoloModelInfo.input_w_ = 608;
    yoloModelInfo.input_h_ = 608;
    // The format of the input
    yoloModelInfo.input_format_ = "FORMAT_NCHW";
    yoloModelInfo.type1_ = CV_32FC1;
    yoloModelInfo.type3_ = CV_32FC3;
    yoloModelInfo.max_batch_size_ = 32;
    yoloModelInfo.shape_.push_back(batch_size);
    yoloModelInfo.shape_.push_back(yoloModelInfo.input_c_);
    yoloModelInfo.shape_.push_back(yoloModelInfo.input_h_);
    yoloModelInfo.shape_.push_back(yoloModelInfo.input_w_);

}

int tritonServerInit(Param FLAGS){
    TritonClient triton_union;
    tc::Error t_client = tc::InferenceServerGrpcClient::Create(&triton_union.grpc_client_, FLAGS.url, false);
    if (t_client.IsOk());
    else 
    {
        std::cout << "Error Message: " << t_client.Message() << std::endl;
    }

    bool bptr = false;
    
    if (triton_union.grpc_client_ -> IsServerLive(&bptr, {}).IsOk());
    else {
        std::cerr << "FAILED : server not alive" << std::endl;
        return 0;
    }
    
    bptr = false;
    if (triton_union.grpc_client_->IsServerReady(&bptr, {}).IsOk());
    else {
        std::cerr << "FAILED : server not ready" << std::endl;
        return 0;
    }

    bool isModelReady = false;
    if (triton_union.grpc_client_->IsModelReady(&isModelReady, FLAGS.model, "", {}).IsOk());
    else{
        std::cerr << "FAILED : model not ready" << std::endl;
        return 0;
    }


    inference::ModelMetadataResponse model_metadata;
    inference::ModelConfigResponse model_config;

    if (triton_union.grpc_client_->ModelMetadata(&model_metadata, FLAGS.model).IsOk());
    else{
        std::cerr << "FAILED : no model metadata" << std::endl;
        return 0;
    }

    if (triton_union.grpc_client_->ModelConfig(&model_config, FLAGS.model).IsOk());
    else{
        std::cerr << "FAILED : no model config" << std::endl;
        return 0;
    }
    return 1;
}


std::vector<uint8_t> Preprocess(
    const cv::Mat& img, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size)
{
    // Image channels are in BGR order. Currently model configuration
    // data doesn't provide any information as to the expected channel
    // orderings (like RGB, BGR). We are going to assume that RGB is the
    // most likely ordering and so change the channels to that ordering.
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
    cv::Mat sample_resized;
    if (sample.size() != img_size)
    {
        cv::resize(sample, sample_resized, img_size);
    }
    else
    {
        sample_resized = sample;
    }

    cv::Mat sample_type;
    sample_resized.convertTo(
        sample_type, (img_channels == 3) ? img_type3 : img_type1);

    cv::Mat sample_final;
    sample.convertTo(
        sample_type, (img_channels == 3) ? img_type3 : img_type1);
    const int INPUT_W = 640;//Yolo::INPUT_W;
    const int INPUT_H = 640;//Yolo::INPUT_H;
    int w, h, x, y;
    float r_w = INPUT_W / (sample_type.cols * 1.0);
    float r_h = INPUT_H / (sample_type.rows * 1.0);
    if (r_h > r_w)
    {
        w = INPUT_W;
        h = r_w * sample_type.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    }
    else
    {
        w = r_h * sample_type.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(sample_type, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    out.convertTo(sample_final, CV_32FC3, 1.f / 255.f);


    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample_final.total() * sample_final.elemSize();
    size_t pos = 0;
    input_data.resize(img_byte_size);

    // (format.compare("FORMAT_NCHW") == 0)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'input_data'. The BGR channels are
    // backed by the 'input_data' vector so that ends up with CHW
    // order of the data.
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i)
    {
        input_bgr_channels.emplace_back(
            img_size.height, img_size.width, img_type1, &(input_data[pos]));
        pos += input_bgr_channels.back().total() *
            input_bgr_channels.back().elemSize();
    }

    cv::split(sample_final, input_bgr_channels);

    if (pos != img_byte_size)
    {
        std::cerr << "unexpected total size of channels " << pos << ", expecting "
            << img_byte_size << std::endl;
        exit(1);
    }

    return input_data;
}






auto PostprocessYoloV4(
    tc::InferResult* result,
    const size_t batch_size,
    const std::vector<std::string>& output_names, const bool batching)
{
    if (!result->RequestStatus().IsOk())
    {
        std::cerr << "inference  failed with error: " << result->RequestStatus()
            << std::endl;
        exit(1);
    }

    std::vector<float> detections;
    std::vector<int64_t> shape;


    float* outputData;
    size_t outputByteSize;
    for (auto outputName : output_names)
    {
        if (outputName == "prob")
        { 
            result->RawData(
                outputName, (const uint8_t**)&outputData, &outputByteSize);

            tc::Error err = result->Shape(outputName, &shape);
            detections = std::vector<float>(outputByteSize / sizeof(float));
            std::memcpy(detections.data(), outputData, outputByteSize);
            if (!err.IsOk())
            {
                std::cerr << "unable to get data for " << outputName << std::endl;
                exit(1);
            }
        }

    }

    return make_tuple(detections, shape);
}


void detect(const cv::Mat& image) {
    Param FLAGS;
    // TritonClient triton_union;
    // tc::Error client = tc::InferenceServerGrpcClient::Create(&triton_union.grpc_client_, FLAGS.url, false);

    tc::InferInput* input;
    tc::Error err;
    err = tc::InferInput::Create(&input, "input_name", {1, 3, FLAGS.height, FLAGS.width}, "FP32");
    if(!err.IsOk()){
        std::cerr << "unable to get input: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);
    std::vector<tc::InferInput *> inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput *> outputs;

    tc::InferRequestedOutput *output;
    err =
        tc::InferRequestedOutput::Create(&output, "0");
    if (!err.IsOk())
    {
        std::cerr << "unable to get output: " << err << std::endl;
        exit(1);
    }
    else std::cout << "Created output " << "0" << std::endl;
    outputs.push_back(std::move(output));

    tc::InferOptions options(FLAGS.model);
    options.model_version_ = "";

    tc::InferResult *result;
    std::unique_ptr<tc::InferResult> result_ptr;
    


    // 将图像数据添加到 InferInput
    // 这取决于 Triton 模型的输入要求，可能需要将图像数据从 cv::Mat 转换为字节数组
    // input->AppendRaw(image_data, byte_size);

    // 创建 InferRequestedOutput 对象来表示输出数据
    //tc::InferRequestedOutput* output;
    //tc::InferRequestedOutput::Create(&output, "output_name");

    // 创建 InferRequest 对象并将输入和输出添加到请求中
    //tc::InferRequest request;
    //request.AddInput(input);
    //request.AddRequestedOutput(output);

    // 发送推理请求
   // client->   Infer(&request);

    // 等待推理完成
    //request.WaitForCallbacks();

    // 解析推理结果
    //tc::InferResult* result = request.GetResult();
    //std::string model_name;
    //result->ModelName(&model_name);

    // 获取输出结果并进行后处理
    //const uint8_t* output_data;
    //size_t output_byte_size;
    //result->RawData("output_name", &output_data, &output_byte_size);
    // 这里可以对输出数据进行解析和后处理

    // 释放资源
    //delete input;
    //delete output;
    //delete result;

//----------------------------------------------------------------------------------------
    // 执行非极大值抑制（你需要单独实现'nonMaxSuppression'函数）

    //std::vector<BoundingBox> filteredBoxes = nonMaxSuppression(detectedObjects, originH, originW, inputShape.first, inputShape.second, confThres, nmsThres);

    //return 0;//filteredBoxes;
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat ros_image;
    try
    {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        ros_image = cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'BGR8'.", msg->encoding.c_str());
    }

   

    //reshapedImage(imageHeight, imageWidth, CV_8UC3); // 三通道图像

    // 复制图像数据，并进行翻转操作
    cv::flip(ros_image, ros_image, -1);
    detect(ros_image);
    cv::imshow("Image Window", ros_image);
    cv::waitKey(1);

    // 创建一个与Python中np.frombuffer操作相同的数组
    //cv::Mat reshapedImage(imageHeight, imageWidth, CV_8UC(numChannels), ros_image.data);

    // 打印图像的尺寸
    //std::cout<<reshapedImage.size()<<std::endl;
}


int main(int argc, char** argv)
{
    
    Param FLAGS;
    // Get the model metadata
    if (tritonServerInit(FLAGS)){
        std::cout << "**TRITON SERVER READY**" << std::endl;
    }
    else{
        exit(0);
    }

    ros::init(argc, argv, "detection_client");
    ros::NodeHandle nh;

    pub = nh.advertise<geometry_msgs::TransformStamped>("/detect_result_pub", 10);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("/cam0/image_raw", 10, imageCallback);

    ros::spin();
    return 0;
}
