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
      cout << "failed to read img " << endl;
      return false;
    }


  return true;
}

//
bool preprocess(const cv::Mat &img, torch::Tensor &input_tensor)
{


  input_tensor = torch::from_blob(
    img.data, {1, IMG_SIZE, IMG_SIZE, IMG_CHN});
  input_tensor = input_tensor.permute({0, 3, 1, 2});

  input_tensor[0][0] = input_tensor[0][0].sub_(img_normalize_mean[0]).div_(img_normalize_std[0]);
  input_tensor[0][1] = input_tensor[0][1].sub_(img_normalize_mean[1]).div_(img_normalize_std[1]);
  input_tensor[0][2] = input_tensor[0][2].sub_(img_normalize_mean[2]).div_(img_normalize_std[2]);
}


bool postprocess(const torch::Tensor &boxes, vector<vector<Point>> &contours)
{

    float x = (boxes[0][0]).item().toFloat();
    float y = (boxes[0][1]).item().toFloat();
    float w = (boxes[0][2]).item().toFloat();
    float h = (boxes[0][3]).item().toFloat();
    float theta = (boxes[0][4]).item().toFloat() * 180 / M_PI;

    RotatedRect rRect = RotatedRect(Point2f(x,y), Size2f(w,h), theta);
    Point2f vertices[4];      //定义4个点的数组
    rRect.points(vertices);   //将四个点存储到vertices数组中

    // drawContours函数需要的点是Point类型而不是Point2f类型
    vector<Point> contour;
    for (int i=0; i<4; i++)
    {
      contour.emplace_back(Point(vertices[i]));
    }

    contours.push_back(contour);

}


bool LoadBoxes(QString &gt_path, std::vector<std::vector<cv::Point>> &contours)
{
    QFile f(gt_path);
    if(!f.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        cout <<"Can't open the file!"<<endl;
    }

    while(!f.atEnd())
    {
        vector<Point> contour;

        QByteArray line = f.readLine();
        QString line_str(line);

        QStringList str_list = line_str.split(",");
        for(int i=0; i<8; i=i+2)
        {
            int x = str_list[i].toInt();
            int y = str_list[i+1].toInt();
            contour.emplace_back(cv::Point(x,y));
        }

        contours.push_back(contour);
    }

}





