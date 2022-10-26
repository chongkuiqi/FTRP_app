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

    // 保存路径
    connect(ui->bu_browse_save, &QPushButton::clicked, this, &Detection::browse_save);

    // 检测功能
    connect(ui->bu_detect, &QPushButton::clicked, this, &Detection::detect);

    connect(ui->bu_save, &QPushButton::clicked, this, &Detection::save_results);

    // 显示软件日志
    ui->text_log->setText("请输入图像路径和模型路径...");

    // 读取图像文件路径按钮
    connect(ui->bu_browse_opt, &QPushButton::clicked, this, &Detection::browse_img_opt);
    connect(ui->bu_browse_IR, &QPushButton::clicked, this, &Detection::browse_img_IR);
    connect(ui->bu_browse_SAR, &QPushButton::clicked, this, &Detection::browse_img_SAR);

    // 设置label的大小
    int h=512,w=512;
    ui->labelImage_opt->resize(w,h);
    ui->labelImage_IR->resize(w,h);
    ui->labelImage_SAR->resize(w,h);
    ui->labelImage_opt_results->resize(w,h);
    ui->labelImage_IR_results->resize(w,h);
    ui->labelImage_SAR_results->resize(w,h);
//    // 图像自适应label的大小
//    ui->labelImage->setScaledContents(true);


    // 切换单频谱/多频谱目标检测模式
    connect(ui->CB_mode, &QComboBox::currentTextChanged, this, &Detection::switch_mode);

    // 单频谱图像选择按钮，放进一个按钮组
    bu_group_SS.setParent(this);
    // 按钮组中互斥
    bu_group_SS.setExclusive(true);
    bu_group_SS.addButton(ui->RB_opt, 0);
    bu_group_SS.addButton(ui->RB_IR, 1);
    bu_group_SS.addButton(ui->RB_SAR, 2);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_SS, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Detection::on_bu_group_SS);

    // 多频谱图像选择按钮，放进一个按钮组
    bu_group_MS.setParent(this);
    // 按钮组中不互斥
    bu_group_MS.setExclusive(false);
    bu_group_MS.addButton(ui->CB_opt, 0);
    bu_group_MS.addButton(ui->CB_IR, 1);
    bu_group_MS.addButton(ui->CB_SAR, 2);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_MS, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Detection::on_bu_group_MS);


    connect(ui->CB_mode_show, &QComboBox::currentTextChanged, this, &Detection::switch_mode_show);

//    ui->tabWidget_software->setCurrentWidget(ui->software1);

}

Detection::~Detection()
{
    delete ui;
}

void Detection::get_bu_group_status(QButtonGroup *bu_group)
{
    // 状态归零
    this->img_status = {false, false, false};

    // 遍历按钮，获取选中状态
    QList<QAbstractButton*> list = bu_group->buttons();
    foreach (QAbstractButton *pCheckBox, list)
    {
        // 获取按钮名称和状态
       QString bu_name = pCheckBox->text();
       bool status = pCheckBox->isChecked();

       if (bu_name == QString("可见光")) this->img_status.opt = status;
       if (bu_name == QString("红外")) this->img_status.IR = status;
       if (bu_name == QString("SAR")) this->img_status.SAR = status;

    }

    std::cout << "模式：" << std::endl;
    std::cout << "可见光:" << img_status.opt << std::endl;
    std::cout << "红外:" << img_status.IR << std::endl;
    std::cout << "SAR:" << img_status.SAR << std::endl;

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

void Detection::on_bu_group_SS(QAbstractButton *button)
{
    this->get_bu_group_status(&bu_group_SS);
}

void Detection::on_bu_group_MS(QAbstractButton *button)
{
    this->get_bu_group_status(&bu_group_MS);
}

void Detection::switch_mode(const QString &text)
{
    if (text == QString("单频谱图像目标检测"))
    {
        ui->stackedWidget_detect->setCurrentWidget(ui->page_single);
        this->get_bu_group_status(&bu_group_SS);
    }
    else
    {
        ui->stackedWidget_detect->setCurrentWidget(ui->page_MS);
        this->get_bu_group_status(&bu_group_MS);
    }

}

void Detection::switch_mode_show(const QString &text)
{
    if (text == QString("检测前"))
    {
        ui->stackedWidget_show->setCurrentWidget(ui->page_before_show);
    }
    else
    {
        ui->stackedWidget_show->setCurrentWidget(ui->page_after_show);
    }

}

void Detection::browse_img(QString img_type)
{
    QString img_path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Images(*.png *.jpg)"
                );
//    QString path = QFileDialog::getOpenFileName(
//                this,
//                "open",
//                "../",
//                "All(*.pt)"
//                );

    //显示图像
    QImage* srcimg = new QImage;
    srcimg->load(img_path);

    if (img_type==QString("可见光"))
    {
        ui->le_imgpath_opt->setText(img_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("红外"))
    {
        ui->le_imgpath_IR->setText(img_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("SAR"))
    {
        ui->le_imgpath_SAR->setText(img_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
    }


}

void Detection::browse_img_opt()
{
    browse_img(QString("可见光"));

}
void Detection::browse_img_IR()
{
    browse_img(QString("红外"));

}
void Detection::browse_img_SAR()
{
    browse_img(QString("SAR"));

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

    at::Tensor img_tensor;
    std::vector<c10::IValue> imgs;
    // 以下三个if语句，按顺序分别提取可见光、红外、SAR图像，顺序不能乱，否则输入到网络中的图像顺序也是乱的
    if (this->img_status.opt)
    {
        QString img_path = ui->le_imgpath_opt->text();
        LoadImage(img_path.toStdString(), this->img); // CV_8UC3
        cv::Mat img_Opt = this->img.clone();
        preprocess(img_Opt, img_tensor);
        imgs.push_back(img_tensor.clone());
    }
    if (this->img_status.IR)
    {
        QString img_path = ui->le_imgpath_IR->text();
        cv::Mat img_IR;
        LoadImage(img_path.toStdString(), img_IR); // CV_8UC3
        preprocess(img_IR, img_tensor);
        imgs.push_back(img_tensor.clone());
    }
    if (this->img_status.SAR)
    {
        QString img_path = ui->le_imgpath_SAR->text();
        cv::Mat img_SAR;
        LoadImage(img_path.toStdString(), img_SAR); // CV_8UC3
        preprocess(img_SAR, img_tensor);
        imgs.push_back(img_tensor.clone());
    }


    QString img_path = ui->le_imgpath_opt->text();
    //获取图像名称和路径
    QFileInfo imginfo = QFileInfo(img_path);
    // 图像名称
    QString img_name = imginfo.fileName();
    //文件后缀
    QString fileSuffix = imginfo.suffix();
    //绝对路径
    QString filePath = imginfo.absolutePath();




    // 可见光放在百分位上，红外放在十分位上，SAR放在个位上
    torch::NoGradGuard no_grad;
    torch::jit::script::Module model;
    c10::intrusive_ptr<at::ivalue::Tuple> output;
    std::string model_path;
    int num_status = 100*this->img_status.opt + 10*this->img_status.IR + this->img_status.SAR;
    switch(num_status)
    {
        // 可见光
        case 100:
            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/detection/exp361_no_align_script.pt";
//            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/example/FTRP_software/exp369_script.pt";
            break;
//        // 红外
//        case 10:
//            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/detection/exp361_no_align_script.pt";
//            break;
//        // SAR
//        case 1:
//            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/detection/exp361_no_align_script.pt";
//            break;


//        // 可见光+红外
//        case 110:
//            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/detection/exp361_no_align_script.pt";
//            break;
//        // 可见光+SAR
//        case 101:
//            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/detection/exp361_no_align_script.pt";
//            break;
//        // 红外+SAR
//        case 11:
//            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/detection/exp361_no_align_script.pt";
//            break;

        // 可见光+红外+SAR
        case 111:
            model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/detection/exp15_MS_script.pt";
            break;

        default:
            std::cout << "没有选择任何模型！！！" << std::endl;
            break;
    }


    ui->text_log->append("开始推理...");
    model = torch::jit::load(model_path, device);
    output = model.forward(imgs).toTuple();




    c10::List<at::Tensor> scores_levels = output->elements()[0].toTensorList();
    c10::List<at::Tensor> bboxes_levels = output->elements()[1].toTensorList();

    // 获取模型配置参数
    float score_thr = (ui->line_score->text()).toFloat();
    float iou_thr = (ui->line_iou_thr->text()).toFloat();
    int max_before_nms = (ui->line_max_before_nms->text()).toInt();

    det_results results;
    results = NMS(scores_levels, bboxes_levels,
                  max_before_nms=max_before_nms,
                  score_thr=score_thr, iou_thr=iou_thr);

    std::cout << "结束" << std::endl;

    xywhtheta2points(results.boxes, this->contours);

    ui->text_log->append("检测完成！");


    if (this->img_status.opt) this->save_img_opt();
    if (this->img_status.IR) this->save_img_IR();
    if (this->img_status.SAR) this->save_img_SAR();

}


void Detection::save_img(QString img_type)
{

    int contoursIds = -1;
    const cv::Scalar color = cv::Scalar(0,0,255);
    int thickness = 3;

    QString img_path;
    cv::Mat img;
    //获取图像名称和路径
    QFileInfo imginfo;
    // 图像名称
    QString img_name;
    //文件后缀
    QString fileSuffix;
    if (img_type==QString("可见光"))
    {
        img_path = ui->le_imgpath_opt->text();
        LoadImage(img_path.toStdString(), img); // CV_8UC3
        //获取图像名称和路径
        imginfo = QFileInfo(img_path);
        // 图像名称
        img_name = imginfo.fileName();
        //文件后缀
        fileSuffix = imginfo.suffix();

        // 保存图像
        QString save_img_name = img_name;
        save_img_name.replace(QRegExp("."+fileSuffix), QString("_opt."+fileSuffix));
        QString save_path = ui->le_savepath->text() +"/"+ save_img_name;

        drawContours(img, this->contours, contoursIds, color, thickness);
        imwrite(save_path.toStdString(), img);

        //显示图像
        QImage* srcimg = new QImage;
        srcimg->load(save_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_opt_results->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt_results->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("红外"))
    {
        img_path = ui->le_imgpath_IR->text();
        LoadImage(img_path.toStdString(), img); // CV_8UC3
        //获取图像名称和路径
        imginfo = QFileInfo(img_path);
        // 图像名称
        img_name = imginfo.fileName();
        //文件后缀
        fileSuffix = imginfo.suffix();

        // 保存图像
        QString save_img_name = img_name;
        save_img_name.replace(QRegExp("."+fileSuffix), QString("_IR."+fileSuffix));
        QString save_path = ui->le_savepath->text() +"/"+ save_img_name;

        drawContours(img, this->contours, contoursIds, color, thickness);
        imwrite(save_path.toStdString(), img);

        //显示图像
        QImage* srcimg = new QImage;
        srcimg->load(save_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_IR_results->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR_results->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("SAR"))
    {
        img_path = ui->le_imgpath_SAR->text();
        LoadImage(img_path.toStdString(), img); // CV_8UC3
        //获取图像名称和路径
        imginfo = QFileInfo(img_path);
        // 图像名称
        img_name = imginfo.fileName();
        //文件后缀
        fileSuffix = imginfo.suffix();

        // 保存图像
        QString save_img_name = img_name;
        save_img_name.replace(QRegExp("."+fileSuffix), QString("_SAR."+fileSuffix));
        QString save_path = ui->le_savepath->text() +"/"+ save_img_name;

        drawContours(img, this->contours, contoursIds, color, thickness);
        imwrite(save_path.toStdString(), img);

        //显示图像
        QImage* srcimg = new QImage;
        srcimg->load(save_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_SAR_results->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR_results->setPixmap(QPixmap::fromImage(dest));
    }


}

void Detection::save_img_opt()
{
    save_img(QString("可见光"));

}
void Detection::save_img_IR()
{
    save_img(QString("红外"));

}
void Detection::save_img_SAR()
{
    save_img(QString("SAR"));

}


void Detection::save_results()
{
    QString path = ui->le_savepath->text();
    QString img_path;
    if (this->img_status.opt) img_path = ui->le_imgpath_opt->text();
    else if (this->img_status.IR) img_path = ui->le_imgpath_IR->text();
    else if (this->img_status.SAR) img_path = ui->le_imgpath_SAR->text();

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

