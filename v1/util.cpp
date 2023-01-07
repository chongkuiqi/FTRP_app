#include "util.h"
#include <QString>
#include <QFileDialog>
#include <QDebug>

#include <cmath>

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

void preprocess(cv::Mat& img, at::Tensor& input_tensor, float &ratio, int & padding_top, int & padding_left)
{
     // scale image to fit
    int img_h = img.rows;
    int img_w = img.cols;

    float ratio_h = float(IMG_SIZE_H) / float(img_h);
    float ratio_w = float(IMG_SIZE_W) / float(img_w);
    // 根据长度和宽度，选择小缩放比例
    ratio = (ratio_h < ratio_w) ? ratio_h : ratio_w;
    int new_img_h = int(round(img_h * ratio));
    int new_img_w = int(round(img_w * ratio));
    // 等比例缩放
    cv::Size scale(new_img_w, new_img_h);
    cv::Mat img_temp;
    cv::resize(img, img_temp, scale);

    // 计算填充的像素个数
    int dh = IMG_SIZE_H - new_img_h;
    int dw = IMG_SIZE_W - new_img_w;
    dh /= 2;
    dw /= 2;
    // padding
    padding_top = int(round(dh - 0.1));
    int padding_bottom = int(round(dh + 0.1));
    padding_left = int(round(dw - 0.1));
    int padding_right = int(round(dw + 0.1));

    cv::copyMakeBorder(img_temp, img_temp, padding_top, padding_bottom, padding_left, padding_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));




    cv::cvtColor(img_temp, img_temp, cv::COLOR_BGR2RGB);
    // convert [unsigned int] to [float], normalization 1/255
    img_temp.convertTo(img_temp, CV_32FC3);


    input_tensor = torch::from_blob(img_temp.data, { 1, img_temp.rows, img_temp.cols, img_temp.channels() });
    input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
    input_tensor[0][0] = input_tensor[0][0].sub_(img_normalize_mean[0]).div_(img_normalize_std[0]);
    input_tensor[0][1] = input_tensor[0][1].sub_(img_normalize_mean[1]).div_(img_normalize_std[1]);
    input_tensor[0][2] = input_tensor[0][2].sub_(img_normalize_mean[2]).div_(img_normalize_std[2]);

    input_tensor = input_tensor.contiguous();
}

//
//void preprocess(cv::Mat &img, at::Tensor &input_tensor)
//{
//    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//    // // scale image to fit
//    // cv::Size scale(IMG_SIZE, IMG_SIZE);
//    // cv::resize(img2, img2, scale);
//    // convert [unsigned int] to [float]
//    img.convertTo(img, CV_32FC3);

//    input_tensor = torch::from_blob(
//      img.data, {1, IMG_SIZE_H, IMG_SIZE_H, IMG_CHN});
//    input_tensor = input_tensor.permute({0, 3, 1, 2});

//    input_tensor[0][0] = input_tensor[0][0].sub_(img_normalize_mean[0]).div_(img_normalize_std[0]);
//    input_tensor[0][1] = input_tensor[0][1].sub_(img_normalize_mean[1]).div_(img_normalize_std[1]);
//    input_tensor[0][2] = input_tensor[0][2].sub_(img_normalize_mean[2]).div_(img_normalize_std[2]);

//    input_tensor = input_tensor.contiguous();
//}

float norm_angle(float theta)
{
    float range[2] = {-0.25*M_PI, 0.75*M_PI};

    // theta的范围应该是[-pi, pi]
    if (theta<range[0]) theta += M_PI;
    if (theta>range[1]) theta -= M_PI;

    return theta;
}

void points2rbox(const std::vector<cv::Point> & contour, RBox &rbox)
{
    // 根据4个点，找最小旋转矩形
    cv::Point2f pt1 = contour[0];
    cv::Point2f pt2 = contour[1];
    cv::Point2f pt3 = contour[2];
    cv::Point2f pt4 = contour[3];

    float edge1 = sqrt(pow(pt1.x-pt2.x, 2) + pow(pt1.y-pt2.y, 2));
    float edge2 = sqrt(pow(pt2.x-pt3.x, 2) + pow(pt2.y-pt3.y, 2));

    // 确定长短边
    float width, height;
    if (edge1 > edge2)
    {
        width = edge1;
        height = edge2;
    }
    else
    {
        width = edge2;
        height = edge1;
    }


    float theta = 0;
    if (edge1 > edge2) theta = atan2(pt2.y-pt1.y, pt2.x-pt1.x);
    else theta = atan2(pt4.y-pt1.y, pt4.x-pt1.x);
    theta = norm_angle(theta);

    // 转化为角度
    float angle = theta * 180 / M_PI;


    rbox.center = cv::Point2f((pt1.x+pt3.x)/2, (pt1.y+pt3.y)/2);
    rbox.size = cv::Size(width, height);
    rbox.angle = angle;

}

float cal_line_length(cv::Point2f pt1, cv::Point2f pt2)
{
    return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
}

std::vector<cv::Point> get_best_begin_point_single(std::vector<cv::Point> & contour_)
{
    float xmin=contour_[0].x;
    float ymin=contour_[0].y;
    float xmax=contour_[0].x;
    float ymax=contour_[0].y;

    for (int i=0; i<4; i++)
    {
        xmin = (xmin < contour_[i].x) ? xmin : contour_[i].x;
        xmax = (xmax > contour_[i].x) ? xmax : contour_[i].x;

        ymin = (ymin < contour_[i].y) ? ymin : contour_[i].y;
        ymax = (ymax > contour_[i].y) ? ymax : contour_[i].y;
    }

    std::vector<cv::Point> combinate[4];
    combinate[0].push_back(contour_[0]);
    combinate[0].push_back(contour_[1]);
    combinate[0].push_back(contour_[2]);
    combinate[0].push_back(contour_[3]);

    combinate[1].push_back(contour_[1]);
    combinate[1].push_back(contour_[2]);
    combinate[1].push_back(contour_[3]);
    combinate[1].push_back(contour_[0]);

    combinate[2].push_back(contour_[2]);
    combinate[2].push_back(contour_[3]);
    combinate[2].push_back(contour_[0]);
    combinate[2].push_back(contour_[1]);

    combinate[3].push_back(contour_[3]);
    combinate[3].push_back(contour_[0]);
    combinate[3].push_back(contour_[1]);
    combinate[3].push_back(contour_[2]);


    std::vector<cv::Point2f> dst_coordinate;
    dst_coordinate.push_back(cv::Point2f(xmin, ymin));
    dst_coordinate.push_back(cv::Point2f(xmax, ymin));
    dst_coordinate.push_back(cv::Point2f(xmax, ymax));
    dst_coordinate.push_back(cv::Point2f(xmin, ymax));
    float force = 100000000.0;
    int force_flag = 0;
    float temp_force;

    for (int i=0; i<4; i++)
    {
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) +
                cal_line_length(combinate[i][1], dst_coordinate[1]) +
                cal_line_length(combinate[i][2], dst_coordinate[2]) +
                cal_line_length(combinate[i][3], dst_coordinate[3]);

        if (temp_force < force)
        {
            force = temp_force;
            force_flag = i;
        }
    }

    return combinate[force_flag];

}

void rbox2points(std::vector<cv::Point> & contour, const RBox &rbox)
{

    cv::Point2f center = rbox.center;
    float x_c = center.x;
    float y_c = center.y;
    float w = rbox.size.width;
    float h = rbox.size.height;
    float angle = rbox.angle;
    float theta = angle * M_PI / 180;

    float tl_x=-w/2, tl_y=-h/2, br_x=w/2, br_y=h/2;

    cv::Mat rect = (cv::Mat_<float>(2,4) <<
                    tl_x, br_x, br_x, tl_x,
                    tl_y, tl_y, br_y, br_y
                    );
    cv::Mat R = (cv::Mat_<float>(2,2) <<
                 cos(theta), -sin(theta),
                 sin(theta), cos(theta)
                 );

    cv::Mat poly = R * rect;


    float x,y;
    std::vector<cv::Point> contour_;
    for (int i=0; i<4; i++)
    {
        x = poly.at<float>(0,i) + x_c;
        y = poly.at<float>(1,i) + y_c;
        contour_.push_back(cv::Point(x,y));
    }

    contour = get_best_begin_point_single(contour_);

}

void rbox2xywhtheta(const RBox &rbox, at::Tensor &rbox_tensor)
{
    cv::Point2f center = rbox.center;
    float x = center.x;
    float y = center.y;
    float w = rbox.size.width;
    float h = rbox.size.height;

    // 单位是角度
    float angle = rbox.angle;
    // 转化为弧度
    float theta = angle * M_PI / 180;

    rbox_tensor = torch::tensor({x,y,w,h,theta});

}

void points2xywhtheta(const std::vector<cv::Point> & contour, at::Tensor &rbox)
{
    RBox rrbox;
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
    float angle = (rbox[4]).item().toFloat() * 180 / M_PI;

    RBox rRect = {cv::Point2f(x,y), cv::Size2f(w,h), angle};

    rbox2points(contour, rRect);

}

void xywhtheta2points(const at::Tensor &rboxes, std::vector<std::vector<cv::Point>> &contours)
{
    // 先清空所有元素
    contours.clear();

    int num_boxes = rboxes.sizes()[0];

    RBox rRect;
    cv::Point2f vertices[4];      //定义4个点的数组
    // drawContours函数需要的点是Point类型而不是Point2f类型
    std::vector<cv::Point> contour;
    for (int i=0; i<num_boxes; i++)
    {
        float x = (rboxes[i][0]).item().toFloat();
        float y = (rboxes[i][1]).item().toFloat();
        float w = (rboxes[i][2]).item().toFloat();
        float h = (rboxes[i][3]).item().toFloat();
        float angle = (rboxes[i][4]).item().toFloat() * 180 / M_PI;

        rRect = {cv::Point2f(x,y), cv::Size2f(w,h), angle};

        contour.clear();
        rbox2points(contour, rRect);

        contours.push_back(contour);

    }
}




//void points2rbox(const std::vector<cv::Point> & contour, cv::RotatedRect &rbox)
//{
//    // 根据4个点，找最小旋转矩形
//    rbox = cv::minAreaRect(contour);
//}
//void rbox2points(std::vector<cv::Point> & contour, const cv::RotatedRect &rbox)
//{

//    cv::Point2f vertices[4];      //定义4个点的数组
//    rbox.points(vertices);   //将四个点存储到vertices数组中
//    for (int i=0; i<4; i++)
//    {
//      contour.emplace_back(cv::Point(vertices[i]));
//    }
//}

//void points2xywhtheta(const std::vector<cv::Point> & contour, at::Tensor &rbox)
//{
//    cv::RotatedRect rrbox;
//    points2rbox(contour, rrbox);

//    cv::Point2f center = rrbox.center;
//    float x = center.x;
//    float y = center.y;
//    float w = rrbox.size.width;
//    float h = rrbox.size.height;

//    // 单位是角度
//    float angle = rrbox.angle;
//    // 转化为弧度
//    float theta = angle * M_PI / 180;

//    rbox = torch::tensor({x,y,w,h,theta});

//}

//void xywhtheta2points(std::vector<cv::Point> & contour, const at::Tensor &rbox)
//{
//    float x = (rbox[0]).item().toFloat();
//    float y = (rbox[1]).item().toFloat();
//    float w = (rbox[2]).item().toFloat();
//    float h = (rbox[3]).item().toFloat();
//    float theta = (rbox[4]).item().toFloat() * 180 / M_PI;

//    cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(x,y), cv::Size2f(w,h), theta);
//    cv::Point2f vertices[4];      //定义4个点的数组
//    //将四个点存储到vertices数组中
//    rRect.points(vertices);


//    // 先清空所有元素
//    contour.clear();
//    for (int i=0; i<4; i++)
//    {
//      contour.emplace_back(cv::Point2f(vertices[i]));
//    }

//}

//bool xywhtheta2points(const at::Tensor &rboxes, std::vector<std::vector<cv::Point>> &contours)
//{
//    // 先清空所有元素
//    contours.clear();

//    int num_boxes = rboxes.sizes()[0];

//    cv::RotatedRect rRect;
//    cv::Point2f vertices[4];      //定义4个点的数组
//    // drawContours函数需要的点是Point类型而不是Point2f类型
//    std::vector<cv::Point> contour;
//    for (int i=0; i<num_boxes; i++)
//    {
//        float x = (rboxes[i][0]).item().toFloat();
//        float y = (rboxes[i][1]).item().toFloat();
//        float w = (rboxes[i][2]).item().toFloat();
//        float h = (rboxes[i][3]).item().toFloat();
//        float theta = (rboxes[i][4]).item().toFloat() * 180 / M_PI;

//        rRect = cv::RotatedRect(cv::Point2f(x,y), cv::Size2f(w,h), theta);
//        //将四个点存储到vertices数组中
//        rRect.points(vertices);


//        contour.clear();
//        for (int i=0; i<4; i++)
//        {
//          contour.emplace_back(cv::Point(vertices[i]));
//        }
//        contours.push_back(contour);

//    }

//}


void LoadBoxes(QString &gt_path, std::vector<std::vector<cv::Point>> &contours)
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

//void draw_rboxes(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours,
//                 int contoursIds=-1, cv::Scalar color = cv::Scalar(0,0,255), int thickness=3)
void draw_rboxes(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours,
                 int contoursIds, cv::Scalar color, int thickness)
{
    img_result = img.clone();
    // 画前景区域的框
    cv::drawContours(img_result, contours, contoursIds, color, thickness);
}

int get_best_point_for_print(const std::vector<cv::Point> &contour)
{
    float x[4], y[4];
    float y_min=contour[0].y;
    int id = 0;
    for (int i=0; i<4; i++)
    {
        x[i] = contour[i].x;
        y[i] = contour[i].y;

        if (y[i] < y_min)
        {
            y_min = y[i];
            id = i;
        }
    }

    return id;

}
//void draw_rboxes_ids(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours,
//                     cv::Scalar color = cv::Scalar(0,0,255), int thickness=3)
void draw_rboxes_ids(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours,
                     cv::Scalar color, int thickness)
{

    img_result = img.clone();

    int num_boxes = contours.size();
    for (int id=0; id < num_boxes; id++)
    {
        // 画一个框
        color = cv::Scalar(0,0,255); // 编号为红色
        cv::drawContours(img_result, contours, id, color, thickness);


//        putText( InputOutputArray img, const String& text, Point org,
//                                 int fontFace, double fontScale, Scalar color,
//                                 int thickness = 1, int lineType = LINE_8,
//                                 bool bottomLeftOrigin = false )
        // 画出boxes的编号
        std::string box_id = std::to_string(id);

        // 找出四个角点中位置最合适的。这里选择右上角点
        int point_id = get_best_point_for_print(contours[id]);
        cv::Point point = contours[id][point_id];
        int fontFace = cv::FONT_HERSHEY_COMPLEX;        // 字体类型
        double fontScale=2.0;    // 字体大小
        color = cv::Scalar(0,255,255); // 编号为绿色
        cv::putText(img_result, box_id, point, fontFace, fontScale, color, thickness);
    }

}

