#include "util.h"

#include "detection.h"
#include "ui_detection.h"

#include <QFileDialog>
#include <QDebug>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

#include <dlfcn.h>

#include <fstream>

#include "nms_rotated.h"
#include "nms.h"
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

    // 保存路径
    connect(ui->bu_browse_save, &QPushButton::clicked, this, &Detection::browse_save);

    // 检测功能
    connect(ui->bu_detect, &QPushButton::clicked, this, &Detection::detect);

    connect(ui->bu_save, &QPushButton::clicked, this, &Detection::save_results);

    // 显示软件日志
    ui->text_log->setText("请输入图像路径和模型路径...");


    // 设置label的大小
    ui->labelImage->resize(512,512);
    // 图像自适应label的大小
    ui->labelImage->setScaledContents(true);

}

Detection::~Detection()
{
    delete ui;
}

void Detection::browse_img()
{
    QString img_path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Images(*.png *.jpg)"
                );

    // qDebug() << path;
    ui->le_imgpath->setText(img_path);

    //显示图像
    QImage* srcimg = new QImage;
    srcimg->load(img_path);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg->scaled(ui->labelImage->size(),Qt::KeepAspectRatio);
    ui->labelImage->setPixmap(QPixmap::fromImage(dest));

}

void Detection::browse_model()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "All(*.pt)"
                );

    // qDebug() << path;
    ui->le_modelpath->setText(path);
}

void Detection::browse_save()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    // qDebug() << path;
    ui->le_savepath->setText(path);
}

void Detection::detect()
{
//    torch::Device device(torch::kCUDA, 0);
    torch::Device device(torch::kCPU);
    std::cout << "cuda is_available:" << torch::cuda::is_available() << std::endl;

    ui->text_log->append("正在获取图像和模型...");

    QString img_path = ui->le_imgpath->text();
    //获取图像名称和路径
    QFileInfo imginfo = QFileInfo(img_path);
    // 图像名称
    QString img_name = imginfo.fileName();
    //文件后缀
    QString fileSuffix = imginfo.suffix();
//    //绝对路径
//    QString filePath = imginfo.absolutePath();


    LoadImage(img_path.toStdString(), this->img); // CV_8UC3
    cv::Mat img_Opt = this->img.clone();
    // cout << img.at<float>(0,0,0) << endl;
    at::Tensor input_tensor;
    preprocess(img_Opt, input_tensor);
    input_tensor = input_tensor.to(device);



    // cout << img.itemsize() << endl;
//    cout << "input img size:" << input_tensor.sizes() << endl;

    std::string model_path = ui->le_modelpath->text().toStdString();

    torch::NoGradGuard no_grad;
    torch::jit::script::Module model;
    model = torch::jit::load(model_path, device);


    ui->text_log->append("开始推理...");


    c10::intrusive_ptr<at::ivalue::Tuple> output;
//    output = model.forward(inputs).toTuple();
    if (ui->bu_MS_detection->isChecked())
    {

        QString img_IR_path = img_path;
        img_IR_path.replace(QRegExp("."+fileSuffix), "_IR."+fileSuffix).toStdString();
        cv::Mat img_IR;
        LoadImage(img_IR_path.toStdString(), img_IR); // CV_8UC3
        at::Tensor input_tensor_IR;
        preprocess(img_IR, input_tensor_IR);
        input_tensor_IR = input_tensor_IR.to(device);

        QString img_SAR_path = img_path;
        img_SAR_path.replace(QRegExp("."+fileSuffix), "_SAR."+fileSuffix).toStdString();
        cv::Mat img_SAR;
        LoadImage(img_SAR_path.toStdString(), img_SAR); // CV_8UC3
        at::Tensor input_tensor_SAR;
        preprocess(img_SAR, input_tensor_SAR);
        input_tensor_SAR = input_tensor_SAR.to(device);

        output = model.forward({input_tensor, input_tensor_IR, input_tensor_SAR}).toTuple();
    }
    else
    {
        output = model.forward({input_tensor}).toTuple();
    }

    c10::List<at::Tensor> scores_levels = output->elements()[0].toTensorList();
    c10::List<at::Tensor> bboxes_levels = output->elements()[1].toTensorList();

    // 获取模型配置参数
    float score_thr = (ui->line_score->text()).toFloat();
    float iou_thr = (ui->line_iou_thr->text()).toFloat();
    int max_per_img = (ui->line_maxnum->text()).toInt();
    int max_before_nms = (ui->line_max_before_nms->text()).toInt();

    det_results results;
    results = NMS(scores_levels, bboxes_levels,
                  max_before_nms=max_before_nms, score_thr=score_thr,
                  iou_thr=iou_thr, max_per_img=max_per_img);
    std::cout << "结束" << std::endl;

    xywhtheta2xywh4points(results.boxes, this->contours);

    ui->text_log->append("检测完成！");


    int contoursIds = -1;
    const cv::Scalar color = cv::Scalar(0,0,255);
    int thickness = 3;
    this->img_result = this->img.clone();
    drawContours(this->img_result, this->contours, contoursIds, color, thickness);

    // 保存图像
    QString save_path = ui->le_savepath->text() +"/"+ img_name;
    imwrite(save_path.toStdString(), this->img_result);


    //显示图像
    QImage* srcimg = new QImage;
    srcimg->load(save_path);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg->scaled(ui->labelImage->size(),Qt::KeepAspectRatio);
    ui->labelImage->setPixmap(QPixmap::fromImage(dest));

}

void Detection::save_results()
{
    QString path = ui->le_savepath->text();
    QString img_path = ui->le_imgpath->text();
    //获取图像名称和路径
    QFileInfo imginfo = QFileInfo(img_path);
    // 图像名称
    QString img_name = imginfo.fileName();
    //文件后缀
    QString fileSuffix = imginfo.suffix();

    // 保存检测结果
    QString txt_name = img_name;
    txt_name.replace(QRegExp("."+fileSuffix), QString(".txt"));
    QString txt_save_path = path +"/"+ txt_name;

    std::ofstream fout;
    fout.open(txt_save_path.toStdString());
    int num_boxes = this->contours.size();

    std::string s, line;
    for (int i=0; i<num_boxes; i++)
    {
        s.clear();
        line.clear();
        for (int j=0; j<4; j++)
        {
            int x = this->contours[i][j].x;
            int y = this->contours[i][j].y;

            if (j<3) s = std::to_string(x) + ',' + std::to_string(y) + ',';
            else s = std::to_string(x) + ',' + std::to_string(y) + '\n';

            line = line + s;
        }
        fout << line;
    }

    fout.close();

}

