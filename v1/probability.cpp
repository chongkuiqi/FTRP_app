#include "util.h"
#include "probability.h"
#include "ui_probability.h"
#include <QFileDialog>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <util.h>
#include <fstream>

#include <string>
using namespace std;
using namespace cv;

#include <QDebug>

Probability::Probability(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Probability)
{
    ui->setupUi(this);
    connect(ui->bu_exit, &QPushButton::clicked, this, &QMainWindow::close);
    // 读取图像文件路径
    connect(ui->bu_browse, &QPushButton::clicked, this, &Probability::browse_img);

    // 读取模型文件路径
    connect(ui->bu_browse_gt, &QPushButton::clicked, this, &Probability::browse_gt);

    // 提取特征
    connect(ui->bu_extract, &QPushButton::clicked, this, &Probability::extract_fe);


    connect(ui->bu_similarity, &QPushButton::clicked, this, &Probability::cal_similarity);

    connect(ui->bu_probability, &QPushButton::clicked, this, &Probability::cal_probability);

    connect(ui->bu_save, &QPushButton::clicked, this, &Probability::save_results);
    // 显示软件日志
    ui->text_log->setText("请输入图像路径和标注文件路径...");



    // 三种图像特征选择按钮，放进一个按钮组
    bu_fe_group.setParent(this);
    // 按钮组中不互斥
    bu_fe_group.setExclusive(false);
    bu_fe_group.addButton(ui->bu_deep_fe, 0);
    bu_fe_group.addButton(ui->bu_gray_fe, 1);
    bu_fe_group.addButton(ui->bu_text_fe, 2);

    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_fe_group, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_fe_group);

    fe_status = {false, false, false};


}

Probability::~Probability()
{
    delete ui;
}


void Probability::browse_img()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Images(*.png *.jpg)"
                );

    ui->le_imgpath->setText(path);
}

void Probability::browse_gt()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Text files(*.txt)"
                );

    ui->le_gtpath->setText(path);
}



void Probability::extract_fe()
{
    QString img_path = ui->le_imgpath->text();

    cv::Mat img;
    LoadImage(img_path.toStdString(), img); // CV_8UC3


    // 读取标签文件
    QString gt_path = ui->le_gtpath->text();
    std::vector<std::vector<cv::Point>> contours;
    LoadBoxes(gt_path, contours);



    // 选择特征
    if (this->fe_status.deep) extract_fe_deep(img, contours);
    if (this->fe_status.gray) extract_fe_gray(img, contours);
    if (this->fe_status.text) extract_fe_texture(img, contours);


////    ui->CB
//    QString fe_type = ui->CB_fe->currentText();


//    if (fe_type == QString("深度学习特征")) extract_fe_deep(img, contours);
//    else if (fe_type == QString("灰度特征")) extract_fe_gray(img, contours);
//    else if (fe_type == QString("纹理特征")) extract_fe_texture(img, contours);
//    else cout << "必须选择特征！！！" << endl;

}


void Probability::extract_fe_deep(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{
    std::string model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/example/FTRP_software/exp361_no_align_extract_fe_script.pt";

    torch::Device device(torch::kCPU);
    cv::Mat img2;
    cv::cvtColor(img, img2, cv::COLOR_BGR2RGB);
    // // scale image to fit
    // cv::Size scale(IMG_SIZE, IMG_SIZE);
    // cv::resize(img2, img2, scale);
    // convert [unsigned int] to [float]
    img2.convertTo(img2, CV_32FC3);

    // cout << img.at<float>(0,0,0) << endl;
    //read image tensor
    torch::Tensor input_tensor;
    preprocess(img2, input_tensor);

    input_tensor = input_tensor.to(device);
    // cout << "input img size:" << input_tensor.sizes() << endl;


    torch::NoGradGuard no_grad;
    torch::jit::script::Module module;
    module = torch::jit::load(model_path, device);

    ui->text_log->append("开始推理...");
    auto output = module.forward({input_tensor}).toTuple();

    auto ele = output->elements();
    for (int i=0; i<3; i++)
    {
        cout << "特征层级" << i << "尺寸:" << ele[i].toTensor().sizes() << endl;
    }

    ui->text_log->append("获得深度学习特征！");


}

void Probability::extract_fe_gray(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{
    // 根据4个点，找最小旋转矩形
    cv::RotatedRect box = cv::minAreaRect(contours[0]);
    cv::Point2f center = box.center;
    float angle = box.angle;

    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat img_rotate;
    cv::warpAffine(img, img_rotate, M, cv::Size(img.cols, img.rows), INTER_LINEAR, 0);


//    QFileInfo imginfo = QFileInfo(img_path);
//    QString fileSuffix = imginfo.suffix();

//    QString save_path = img_path.replace(QRegExp("."+fileSuffix), "_rotate."+fileSuffix);
//    cv::imwrite(save_path.toStdString(), img_rotate);

    int x = (int)(center.x);
    int y = (int)(center.y);
    int w = (int)(box.size.width);
    int h = (int)(box.size.height);
    int x1 = x - w/2;
    int y1 = y - h/2;
    // 左上角点坐标，w,h
    cv::Rect rect(x1,y1,w,h);
    cv::Mat roi = img_rotate(rect);

//    QString crop_path = save_path.replace(QRegExp("."+fileSuffix), "_crop."+fileSuffix);
//    cv::imwrite((crop_path).toStdString(), roi);

    // 转化为灰度图
//    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);
//    std::vector<Mat>bgr_plane;
//    cv::split(roi, bgr_plane);
    cv::Mat roi_hist;
    get_hist(roi, roi_hist);

    int padding = (ui->le_bg_ratio->text().toFloat()) * w / 2;
    int x2 = x - w/2 - padding;
    int y2 = y - h/2 - padding;
    // 左上角点坐标，w,h
    cv::Rect rect2(x2,y2,w+2*padding,h+2*padding);
    cv::Mat bg = img_rotate(rect2);
    cv::Mat bg_hist;
    get_hist(bg, bg_hist);

    double roimaxVal=0, bgmaxVal=0;
    bg_hist = bg_hist - roi_hist;
    minMaxLoc(roi_hist, 0, &roimaxVal, 0, 0);
    minMaxLoc(bg_hist, 0, &bgmaxVal, 0, 0);

    roi_hist = roi_hist / roimaxVal;
    bg_hist = bg_hist / bgmaxVal;

    this->roi_fe.gray = roi_hist.clone();
    this->bg_fe.gray = bg_hist.clone();

    ui->text_log->append("获得灰度特征！");


    // 把两个框都画出来

    int contoursIds = -1;
    Scalar color = Scalar(0,0,255);
    int thickness = 3;
    cv::Mat img_show = img.clone();
    // 画前景区域的框
    drawContours(img_show, contours, contoursIds, color, thickness);

    std::vector<std::vector<cv::Point>> bg_contours;
    RotatedRect rRect = RotatedRect(Point2f(x,y), Size2f(w+2*padding, h+2*padding), angle);
    Point2f vertices[4];      //定义4个点的数组
    rRect.points(vertices);   //将四个点存储到vertices数组中
    // drawContours函数需要的点是Point类型而不是Point2f类型
    vector<Point> contour;
    for (int i=0; i<4; i++)
    {
      contour.emplace_back(Point(vertices[i]));
    }
    bg_contours.push_back(contour);

    // 画北背景区域的框
    color = Scalar(255,0,0);
    drawContours(img_show, bg_contours, contoursIds, color, thickness);

    imshow( "image", img_show );
    waitKey(0);


}

void Probability::extract_fe_texture(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{

    ui->text_log->append("获得纹理特征！");
}


void Probability::get_hist(cv::Mat & img, cv::Mat & hist)
{
    //    //定义参数变量
        int bins = 8;
        int histSize[] = {bins, bins};
        float bin_range[] = { 0, 255 };
        const float* ranges[] = { bin_range, bin_range };
        int channels[] = {0, 1};

        // 计算得到直方图数据
        calcHist( &img, 1, channels, Mat(), // do not use mask
                 hist, 2, histSize, ranges,
                 true, // the histogram is uniform
                 false );
        /*
        参数解析：
        all_channel[i]:传入要计算直方图的通道，根据函数原函数可以得出，要以引用的方式传入
        1：传入图像的个数，就一个
        0：表示传入一个通道
        Mat():没有定义掩膜，所以默认计算区域是全图像
        b/g/r_hist:用来存储计算得到的直方图数据
        1：对于当前通道需要统计的直方图个数，我们统计一个
        bin:直方图的横坐标有多少个，我们将其赋值为256，即统计每一个像素值的数量。要求用引用方式传入。
        ranges：每个像素点的灰度等级，要求以引用方式传入。
        false：进行归一化，
        false：计算多个图像的直方图时，不累加上一张图像的像素点数据。
        */
}


void Probability::cal_similarity()
{
    // 首先把所有特征相似度归0
    this->fe_similarity = {0.0, 0.0, 0.0};

    if (this->fe_status.deep) cal_similarity_deep();
    if (this->fe_status.gray) cal_similarity_gray();
    if (this->fe_status.text) cal_similarity_text();

    // 根据特征相似度计算综合相似度
    this->similarity = this->fe_similarity.deep +
            this->fe_similarity.gray +
            this->fe_similarity.text;


}


void Probability::cal_similarity_deep()
{

}


void Probability::cal_similarity_gray()
{
    cv::Mat diff = cv::abs(this->roi_fe.gray - this->bg_fe.gray);

    // cv::mean的返回结果有4个通道
    this->similarity = 1 - cv::mean(diff).val[0];

    QString log = QString("\n计算得到的特征相似度为:") + QString::number(this->similarity, 'f', 3);
    ui->text_log->append(log);
}


void Probability::cal_similarity_text()
{

}

void Probability::cal_probability()
{
    this->probability = 1 - this->similarity;
    QString log = QString("\n计算得到的识别概率为:") + QString::number(this->probability, 'f', 3);
    ui->text_log->append(log);
}

void Probability::save_results()
{
    QString img_path = ui->le_imgpath->text();

    QFileInfo imginfo = QFileInfo(img_path);
    QString img_name = imginfo.fileName();
    QString fileSuffix = imginfo.suffix();

    QString txt_save_path = img_path.replace(QRegExp("."+fileSuffix), "_probability.txt");

    ofstream fout;
    fout.open(txt_save_path.toStdString());

    std::string s = "特征相似度:" + std::to_string(this->similarity) + "\n";
    fout << s;

    s = "目标识别概率:" + std::to_string(this->probability) + "\n";
    fout << s;

    fout.close();

    ui->text_log->append("已完成保存！！！");


}


void Probability::on_bu_fe_group(QAbstractButton *button)
{

    bool status = button->isChecked() ? true : false;
    if (button->text() == QString("深度学习特征")) this->fe_status.deep = status;
    if (button->text() == QString("灰度特征")) this->fe_status.gray = status;
    if (button->text() == QString("纹理特征")) this->fe_status.text = status;

        // 当前点击的按钮
    //    qDebug() << QString("Clicked Button : %1").arg(button->text());

        // 遍历按钮，获取选中状态
//    QList<QAbstractButton*> list = bu_fe_group.buttons();
//    foreach (QAbstractButton *pCheckBox, list)
//    {
//       QString strStatus = pCheckBox->isChecked() ? "Checked" : "Unchecked";
//       qDebug() << QString("Button : %1 is %2").arg(pCheckBox->text()).arg(strStatus);
//    }


}
