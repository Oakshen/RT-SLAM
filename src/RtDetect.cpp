//
// Created by yuwenlu on 2022/3/14.
//
#include <RtDetect.h>

RtDetection::RtDetection()
{
    // 初始化 RtDetrUltralytics 和 TRTInfer
    detector = std::make_unique<RtDetrUltralytics>(0.4, 640, 640); // 设置置信度阈值和网络尺寸
    engine = std::make_unique<TRTInfer>("rtdetr-EVit.engine");


    std::ifstream f("coco.names");
    std::string name = "";
    //读取 coco 数据集类别名称
    while (std::getline(f, name))
    {
        mClassnames.push_back(name);//存储在 mClassnames 向量中
    }
    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear","sports ball"};
}

RtDetection::~RtDetection()//析构函数
{

}

bool RtDetection::Detect()
{
    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }

    cv::Mat img;

    

    // Preparing input tensor
    cv::resize(mRGB, img, cv::Size(640, 640));//mRGB 是原始图像 img 是调整大小后的图像
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);//从BGR 颜色空间转换为 RGB 颜色空间
    const auto input_blob = detector->preprocess_image(img);
    const auto [outputs, shapes] = engine->get_infer_results(input_blob);
    std::vector<Detection> detections = detector->postprocess(outputs, shapes, img.size());

    if (!detections.empty())
    {
        for (const auto& d : detections)
        {
            float left = d.bbox.x * mRGB.cols / 640;
            float top = d.bbox.y * mRGB.rows / 640;
            float right = (d.bbox.x + d.bbox.width) * mRGB.cols / 640;
            float bottom = (d.bbox.y + d.bbox.height) * mRGB.rows / 640;
            int classID = d.label;

            cv::Rect2i DetectArea(left, top, (right - left), (bottom - top));
            mmDetectMap[mClassnames[classID]].push_back(DetectArea);

            if (std::count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID]))
            {
                cv::Rect2i DynamicArea(left, top, (right - left), (bottom - top));
                mvDynamicArea.push_back(DynamicArea);
            }
        }
        if (mvDynamicArea.empty())
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
        }
    }
    return true;
}






void RtDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void RtDetection::ClearImage()
{
    mRGB = cv::Mat();
}

void RtDetection::ClearArea()
{
    mvPersonArea.clear();
    mvDynamicArea.clear();
    mmDetectMap.clear();
}
