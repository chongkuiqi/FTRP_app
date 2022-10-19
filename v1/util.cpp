#include "util.h"
#include <QString>
#include <QFileDialog>

float img_normalize_mean[3] = {123.675, 116.28, 103.53};
float img_normalize_std[3] = {58.395, 57.12, 57.375};

bool LoadImage(std::string file_name, cv::Mat &img)
{
  img = cv::imread(file_name); // CV_8UC3
  if (img.empty() || !img.data)
    {
      std::cout << "failed to read img " << std::endl;
      return false;
    }


  return true;
}

//
bool preprocess(cv::Mat &img, at::Tensor &input_tensor)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // // scale image to fit
    // cv::Size scale(IMG_SIZE, IMG_SIZE);
    // cv::resize(img2, img2, scale);
    // convert [unsigned int] to [float]
    img.convertTo(img, CV_32FC3);

    input_tensor = torch::from_blob(
      img.data, {1, IMG_SIZE, IMG_SIZE, IMG_CHN});
    input_tensor = input_tensor.permute({0, 3, 1, 2});

    input_tensor[0][0] = input_tensor[0][0].sub_(img_normalize_mean[0]).div_(img_normalize_std[0]);
    input_tensor[0][1] = input_tensor[0][1].sub_(img_normalize_mean[1]).div_(img_normalize_std[1]);
    input_tensor[0][2] = input_tensor[0][2].sub_(img_normalize_mean[2]).div_(img_normalize_std[2]);
}


bool xywhtheta2xywh4points(const at::Tensor &boxes, std::vector<std::vector<cv::Point>> &contours)
{
    // 先清空所有元素
    contours.clear();

    int num_boxes = boxes.sizes()[0];

    cv::RotatedRect rRect;
    cv::Point2f vertices[4];      //定义4个点的数组
    // drawContours函数需要的点是Point类型而不是Point2f类型
    std::vector<cv::Point> contour;
    for (int i=0; i<num_boxes; i++)
    {
        float x = (boxes[i][0]).item().toFloat();
        float y = (boxes[i][1]).item().toFloat();
        float w = (boxes[i][2]).item().toFloat();
        float h = (boxes[i][3]).item().toFloat();
        float theta = (boxes[i][4]).item().toFloat() * 180 / M_PI;

        rRect = cv::RotatedRect(cv::Point2f(x,y), cv::Size2f(w,h), theta);
        //将四个点存储到vertices数组中
        rRect.points(vertices);


        contour.clear();
        for (int i=0; i<4; i++)
        {
          contour.emplace_back(cv::Point(vertices[i]));
        }
        contours.push_back(contour);

    }

}



bool LoadBoxes(QString &gt_path, std::vector<std::vector<cv::Point>> &contours)
{
    contours.clear();

    QFile f(gt_path);
    if(!f.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        std::cout <<"Can't open the file!"<< std::endl;
    }

    std::vector<cv::Point> contour;
    while(!f.atEnd())
    {

        QByteArray line = f.readLine();
        QString line_str(line);

        QStringList str_list = line_str.split(",");
        contour.clear();
        for(int i=0; i<8; i=i+2)
        {
            int x = str_list[i].toInt();
            int y = str_list[i+1].toInt();
            contour.emplace_back(cv::Point(x,y));
        }

        contours.push_back(contour);
    }
}





