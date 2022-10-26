#include "util.h"
#include <QString>
#include <QFileDialog>
#include <QDebug>

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
void preprocess(cv::Mat &img, at::Tensor &input_tensor)
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

    input_tensor = input_tensor.contiguous();
}


void points2rbox(const std::vector<cv::Point> & contour, cv::RotatedRect &rbox)
{
    // 根据4个点，找最小旋转矩形
    rbox = cv::minAreaRect(contour);
}
void rbox2points(std::vector<cv::Point> & contour, const cv::RotatedRect &rbox)
{

    cv::Point2f vertices[4];      //定义4个点的数组
    rbox.points(vertices);   //将四个点存储到vertices数组中
    for (int i=0; i<4; i++)
    {
      contour.emplace_back(cv::Point(vertices[i]));
    }

}

void points2xywhtheta(const std::vector<cv::Point> & contour, at::Tensor &rbox)
{
    cv::RotatedRect rrbox;
    points2rbox(contour, rrbox);

    cv::Point2f center = rrbox.center;
    float x = center.x;
    float y = center.y;
    float w = rrbox.size.width;
    float h = rrbox.size.height;

    // 单位是角度
    float angle = rrbox.angle;
    // 转化为弧度
    float theta = angle * M_PI / 180;

    rbox = torch::tensor({x,y,w,h,theta});

}

void xywhtheta2points(std::vector<cv::Point> & contour, const at::Tensor &rbox)
{
    float x = (rbox[0]).item().toFloat();
    float y = (rbox[1]).item().toFloat();
    float w = (rbox[2]).item().toFloat();
    float h = (rbox[3]).item().toFloat();
    float theta = (rbox[4]).item().toFloat() * 180 / M_PI;

    cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(x,y), cv::Size2f(w,h), theta);
    cv::Point2f vertices[4];      //定义4个点的数组
    //将四个点存储到vertices数组中
    rRect.points(vertices);


    // 先清空所有元素
    contour.clear();
    for (int i=0; i<4; i++)
    {
      contour.emplace_back(cv::Point2f(vertices[i]));
    }

}


bool xywhtheta2points(const at::Tensor &rboxes, std::vector<std::vector<cv::Point>> &contours)
{
    // 先清空所有元素
    contours.clear();

    int num_boxes = rboxes.sizes()[0];

    cv::RotatedRect rRect;
    cv::Point2f vertices[4];      //定义4个点的数组
    // drawContours函数需要的点是Point类型而不是Point2f类型
    std::vector<cv::Point> contour;
    for (int i=0; i<num_boxes; i++)
    {
        float x = (rboxes[i][0]).item().toFloat();
        float y = (rboxes[i][1]).item().toFloat();
        float w = (rboxes[i][2]).item().toFloat();
        float h = (rboxes[i][3]).item().toFloat();
        float theta = (rboxes[i][4]).item().toFloat() * 180 / M_PI;

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




QImage MatToImage(const cv::Mat &m)  //Mat转Image
{
    switch(m.type())
    {
        case CV_8UC1:
        {
            QImage img((uchar *)m.data,m.cols,m.rows,m.cols * 1, QImage::Format_Grayscale8);
            return img;
        }
            break;
        case CV_8UC3:
        {
            QImage img((uchar *)m.data,m.cols,m.rows,m.cols * 3, QImage::Format_RGB888);
            return img.rgbSwapped();  //因为在QT中彩色图象是RGB的顺序，但是在OPENCV中是BGR的顺序，所以要转一下
        }
            break;
        case CV_8UC4:
        {
            QImage img((uchar *)m.data,m.cols,m.rows,m.cols * 4, QImage::Format_ARGB32);
            return img;
        }
            break;
        default:     //如果是默认的，那么将其返回为一个空对象
        {
            QImage img;
            return img;
        }
    }
}

QPixmap MatToPixmap(const cv::Mat &m)
{
    return QPixmap::fromImage(MatToImage(m));   //相当于先将Mat转成Image,再转成Pixmap
}

cv::Mat ImageToMat(const QImage &img,bool inCloneImageData)  //Image转Mat
{
    switch(img.format())
    {
        case QImage::Format_Indexed8:   //单通道
        {
            cv::Mat  mat( img.height(), img.width(), CV_8UC1,
                          const_cast<uchar*>(img.bits()), static_cast<size_t>(img.bytesPerLine()) );

            return (inCloneImageData ? mat.clone() : mat);
        }
        // 8-bit, 3 通道
        case QImage::Format_RGB32:   //这种写法表示并列关系
        case QImage::Format_RGB888:
        {
            if ( !inCloneImageData )
            {
                qWarning() << "CVS::QImageToCvMat() - Conversion requires cloning because we use a temporary QImage";
            }

            QImage  swapped = img;

            if ( img.format() == QImage::Format_RGB32 )
            {
                swapped = swapped.convertToFormat( QImage::Format_RGB888 );
            }

            swapped = swapped.rgbSwapped();  //因为在QT中彩色图象是RGB的顺序，但是在OPENCV中是BGR的顺序，所以要转一下

            return cv::Mat( swapped.height(), swapped.width(), CV_8UC3,
                        const_cast<uchar*>(swapped.bits()), static_cast<size_t>(swapped.bytesPerLine()) ).clone();
        }
        // 8-bit, 4 channel
        case QImage::Format_ARGB32:
        case QImage::Format_ARGB32_Premultiplied:
        {
            cv::Mat  mat( img.height(), img.width(), CV_8UC4,
                          const_cast<uchar*>(img.bits()), static_cast<size_t>(img.bytesPerLine()) );

            return (inCloneImageData ? mat.clone() : mat);
        }

        // 8-bit, 1 channel
        default:
            qWarning() << "CVS::QImageToCvMat() - QImage format not handled in switch:" << img.format();
            break;
        }
    return cv::Mat();
}


//void draw_rboxes(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours)
//{
//    int contoursIds=-1;
//    cv::Scalar color = cv::Scalar(0,0,255);
//    int thickness=3;

//    img_result = img.clone();
//    // 画前景区域的框
//    cv::drawContours(img_result, contours, contoursIds, color, thickness);
//}

//void draw_rboxes_ids(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours)
//{
//    cv::Scalar color = cv::Scalar(0,0,255);
//    int thickness=3;

//    img_result = img.clone();

//    int num_boxes = contours.size();
//    for (int id=0; id < num_boxes; id++)
//    {
//        // 画一个框
//        cv::drawContours(img_result, contours, id, color, thickness);


////        putText( InputOutputArray img, const String& text, Point org,
////                                 int fontFace, double fontScale, Scalar color,
////                                 int thickness = 1, int lineType = LINE_8,
////                                 bool bottomLeftOrigin = false )
//        // 画出boxes的编号
//        std::string box_id = std::to_string(id);

//        cv::Point point = contours[id][0];
//        int fontFace = cv::FONT_HERSHEY_COMPLEX;        // 字体类型
//        double fontScale=2.0;    // 字体大小
//        cv::putText(img_result, box_id, point, fontFace, fontScale, color, thickness=thickness);
//    }

//}

