#include "detect.h"

#include "detection.h"
#include "ui_detection.h"

#include <QFileDialog>
#include <QDebug>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>



Detection::Detection(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Detection)
{
    ui->setupUi(this);

    connect(ui->bu_exit, &QPushButton::clicked, this, &QMainWindow::close);
    // 读取图像文件路径
    connect(ui->bu_browse, &QPushButton::clicked, this, &Detection::browse_img);

    // 读取模型文件路径
    connect(ui->bu_browse_model, &QPushButton::clicked, this, &Detection::browse_model);

    connect(ui->bu_detect, &QPushButton::clicked, this, &Detection::detect);

    // 显示软件日志
    ui->text_log->setText("请输入图像路径和模型路径...");

}

Detection::~Detection()
{
    delete ui;
}

void Detection::browse_img()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Images(*.png *.jpg)"
                );

    qDebug() << path;
    ui->le_imgpath->setText(path);
}

void Detection::browse_model()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "All(*.pt)"
                );

    qDebug() << path;
    ui->le_modelpath->setText(path);
}

void Detection::detect()
{
//    torch::Device deviceCUDA(torch::kCUDA, 0);
    torch::Device device(torch::kCPU);

    ui->text_log->append("正在获取图像和模型...");

    std::string img_path = ui->le_imgpath->text().toStdString();
    cv::Mat img;
    LoadImage(img_path, img); // CV_8UC3
    namedWindow( "image", 1 );



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
    // to GPU

//    cout << "cuda is_available:" << torch::cuda::is_available() << endl;

    input_tensor = input_tensor.to(device);
    // cout << img.itemsize() << endl;
//    cout << "input img size:" << input_tensor.sizes() << endl;


    std::string model_path = ui->le_modelpath->text().toStdString();

    torch::NoGradGuard no_grad;
    torch::jit::script::Module module;
    module = torch::jit::load(model_path, device);
    // 获取模型配置参数
    float score_thr = (ui->line_score->text()).toFloat();
    float iou_thr = (ui->line_iou_thr->text()).toFloat();
    int max_per_img = (ui->line_maxnum->text()).toInt();

    cout << score_thr << iou_thr << max_per_img <<endl;
    ui->text_log->append("开始推理...");
    auto output = module.forward({input_tensor}).toTuple();

    // only one img
    auto b = output->elements()[0];
    torch::Tensor boxes = b.toTuple()->elements()[0].toTensor();

    vector<vector<Point>> contours;
    postprocess(boxes, contours);

    ui->text_log->append("检测完成！");
    int contoursIds = -1;
    const Scalar color = Scalar(0,0,255);
    int thickness = 3;
    drawContours(img, contours, contoursIds, color, thickness);
    imshow( "image", img );
    waitKey(0);

}

