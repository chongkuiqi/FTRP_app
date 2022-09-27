#include "detect.h"
#include "probability.h"
#include "ui_probability.h"
#include <QFileDialog>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
using namespace std;
using namespace cv;

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

    connect(ui->bu_extract, &QPushButton::clicked, this, &Probability::extract_fe);

    // 显示软件日志
    ui->text_log->setText("请输入图像路径和模型路径...");

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
    std::string img_path = ui->le_imgpath->text().toStdString();
    std::string gt_path = ui->le_gtpath->text().toStdString();
    std::string model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/example/FTRP_software/exp361_no_align_extract_fe_script.pt";


    torch::Device device(torch::kCPU);

    cv::Mat img;
    LoadImage(img_path, img); // CV_8UC3

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





}
