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


    // 读取图像文件路径按钮
    connect(ui->bu_browse_opt_SS, &QPushButton::clicked, this, &Detection::browse_img_opt_SS);
    connect(ui->bu_browse_IR_SS, &QPushButton::clicked, this, &Detection::browse_img_IR_SS);
    connect(ui->bu_browse_SAR_SS, &QPushButton::clicked, this, &Detection::browse_img_SAR_SS);
    connect(ui->bu_browse_opt_MS, &QPushButton::clicked, this, &Detection::browse_img_opt_MS);
    connect(ui->bu_browse_IR_MS, &QPushButton::clicked, this, &Detection::browse_img_IR_MS);
    connect(ui->bu_browse_SAR_MS, &QPushButton::clicked, this, &Detection::browse_img_SAR_MS);


    // 切换单频谱/多频谱目标检测模式
    connect(ui->CB_mode, &QComboBox::currentTextChanged, this, &Detection::switch_mode);

    // 单频谱图像选择按钮，放进一个按钮组
    bu_group_SS.setParent(this);
    bu_group_SS.addButton(ui->RB_opt, 0);
    bu_group_SS.addButton(ui->RB_IR, 1);
    bu_group_SS.addButton(ui->RB_SAR, 2);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_SS, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Detection::on_bu_group_SS);

    // 多频谱图像选择按钮，放进一个按钮组
    bu_group_MS.setParent(this);
    bu_group_MS.addButton(ui->CB_opt, 0);
    bu_group_MS.addButton(ui->CB_IR, 1);
    bu_group_MS.addButton(ui->CB_SAR, 2);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_MS, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Detection::on_bu_group_MS);


//    ui->tabWidget_software->setCurrentWidget(ui->software1);


    // 初始化各个模块、界面的参数
    this->init_ui();

}

Detection::~Detection()
{
    delete ui;
}

void Detection::get_bu_group_status(QButtonGroup *bu_group, bool is_SS)
{

    this->reset_show();

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

    this->reset_imgpath_show();
    if (is_SS)
    {
        if (this->img_status.opt) ui->widget_opt_SS->show();
        if (this->img_status.IR)  ui->widget_IR_SS->show();
        if (this->img_status.SAR) ui->widget_SAR_SS->show();
    }
    else
    {
        if (this->img_status.opt) ui->widget_opt_MS->show();
        if (this->img_status.IR)  ui->widget_IR_MS->show();
        if (this->img_status.SAR) ui->widget_SAR_MS->show();
    }
}

void Detection::on_bu_group_SS(QAbstractButton *button) {this->get_bu_group_status(&bu_group_SS, true);}

void Detection::on_bu_group_MS(QAbstractButton *button) {this->get_bu_group_status(&bu_group_MS, false);}

void Detection::switch_mode(const QString &text)
{
    this->reset_show();
    this->reset_bu_groups();
    if (text == QString("单频谱图像目标检测"))
    {
        ui->stackedWidget_detect->setCurrentWidget(ui->page_SS);
        this->get_bu_group_status(&bu_group_SS, true);
    }
    else
    {
        ui->stackedWidget_detect->setCurrentWidget(ui->page_MS);
        this->get_bu_group_status(&bu_group_MS,false);
    }

}



void Detection::browse_img(QString img_type, bool is_SS)
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
        this->img_paths.opt = img_path;

        if (is_SS) ui->le_imgpath_opt_SS->setText(img_path);
        else ui->le_imgpath_opt_MS->setText(img_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("红外"))
    {
        this->img_paths.IR = img_path;

        if (is_SS) ui->le_imgpath_IR_SS->setText(img_path);
        else ui->le_imgpath_IR_MS->setText(img_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("SAR"))
    {
        this->img_paths.SAR = img_path;

        if (is_SS) ui->le_imgpath_SAR_SS->setText(img_path);
        else ui->le_imgpath_SAR_MS->setText(img_path);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg->scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
    }

}

void Detection::browse_img_opt_SS() {browse_img(QString("可见光"), true);}
void Detection::browse_img_IR_SS()  {browse_img(QString("红外"), true);}
void Detection::browse_img_SAR_SS() {browse_img(QString("SAR"), true);}
void Detection::browse_img_opt_MS() {browse_img(QString("可见光"), false);}
void Detection::browse_img_IR_MS()  {browse_img(QString("红外"), false);}
void Detection::browse_img_SAR_MS() {browse_img(QString("SAR"), false);}


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
        QString img_path = this->img_paths.opt;
        cv::Mat img_opt;
        LoadImage(img_path.toStdString(), img_opt); // CV_8UC3
        preprocess(img_opt, img_tensor);
        imgs.push_back(img_tensor.clone());
    }
    if (this->img_status.IR)
    {
        QString img_path = this->img_paths.IR;
        std::cout << "opt图像名称：" << this->img_paths.opt.toStdString() << std::endl;
        std::cout << "IR图像名称：" << this->img_paths.IR.toStdString() << std::endl;
        std::cout << "SAR图像名称：" << this->img_paths.SAR.toStdString() << std::endl;
        cv::Mat img_IR;
        LoadImage(img_path.toStdString(), img_IR); // CV_8UC3
        preprocess(img_IR, img_tensor);
        imgs.push_back(img_tensor.clone());
    }
    if (this->img_status.SAR)
    {
        QString img_path = this->img_paths.SAR;
        cv::Mat img_SAR;
        LoadImage(img_path.toStdString(), img_SAR); // CV_8UC3
        preprocess(img_SAR, img_tensor);
        imgs.push_back(img_tensor.clone());
    }



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
            model_path = "./model/detection/可见光_exp361.pt";
            break;
        // 红外
        case 10:
            model_path = "./model/detection/红外_exp382.pt";
            break;
        // SAR
        case 1:
            model_path = "./model/detection/SAR_exp383.pt";
            break;


        // 可见光+红外
        case 110:
            model_path = "./model/detection/可见光_红外_exp17.pt";
            break;
        // 可见光+SAR
        case 101:
            model_path = "./model/detection/可见光_SAR_exp18.pt";
            break;
        // 红外+SAR
        case 11:
            model_path = "./model/detection/红外_SAR_exp19.pt";
            break;

        // 可见光+红外+SAR
        case 111:
            model_path = "./model/detection/可见光_红外_SAR_exp16.pt";
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


    if (this->img_status.opt) this->show_img_opt_results();
    if (this->img_status.IR) this->show_img_IR_results();
    if (this->img_status.SAR) this->show_img_SAR_results();

}


void Detection::show_img_results(QString img_type)
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
        img_path = this->img_paths.opt;
        LoadImage(img_path.toStdString(), img); // CV_8UC3

        drawContours(img, this->contours, contoursIds, color, thickness);

        //显示图像
        QImage srcimg = MatToImage(img);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));

    }

    if (img_type==QString("红外"))
    {
        img_path = this->img_paths.IR;
        LoadImage(img_path.toStdString(), img); // CV_8UC3

        drawContours(img, this->contours, contoursIds, color, thickness);

        //显示图像
        QImage srcimg = MatToImage(img);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));

    }

    if (img_type==QString("SAR"))
    {
        img_path = this->img_paths.SAR;
        LoadImage(img_path.toStdString(), img); // CV_8UC3

        drawContours(img, this->contours, contoursIds, color, thickness);

        //显示图像
        QImage srcimg = MatToImage(img);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
    }


}

void Detection::show_img_opt_results() {show_img_results(QString("可见光"));}
void Detection::show_img_IR_results()  {show_img_results(QString("红外"));}
void Detection::show_img_SAR_results() {show_img_results(QString("SAR"));}

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
        img_path = this->img_paths.opt;
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
        QImage dest = srcimg->scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("红外"))
    {
        img_path = this->img_paths.IR;
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
        QImage dest = srcimg->scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));
    }

    if (img_type==QString("SAR"))
    {
        img_path = this->img_paths.SAR;
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
        QImage dest = srcimg->scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
    }


}

void Detection::save_img_opt() {save_img(QString("可见光"));}
void Detection::save_img_IR()  {save_img(QString("红外"));}
void Detection::save_img_SAR() {save_img(QString("SAR"));}


void Detection::save_results()
{
    // 保存图像
    if (this->img_status.opt) this->save_img_opt();
    if (this->img_status.IR) this->save_img_IR();
    if (this->img_status.SAR) this->save_img_SAR();

    // 保存检测结果
    QString path = ui->le_savepath->text();
    QString img_path;
    if (this->img_status.opt) img_path = this->img_paths.opt;
    else if (this->img_status.IR) img_path = this->img_paths.IR;
    else if (this->img_status.SAR) img_path = this->img_paths.opt;

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


void Detection::reset_show()
{
    this->reset_imgpath_show();
    this->reset_img_show();
}

void Detection::reset_imgpath_show()
{
    ui->le_imgpath_opt_SS->clear();
    ui->le_imgpath_IR_SS->clear();
    ui->le_imgpath_SAR_SS->clear();
    ui->le_imgpath_opt_MS->clear();
    ui->le_imgpath_IR_MS->clear();
    ui->le_imgpath_SAR_MS->clear();

    ui->widget_opt_SS->hide();
    ui->widget_IR_SS->hide();
    ui->widget_SAR_SS->hide();
    ui->widget_opt_MS->hide();
    ui->widget_IR_MS->hide();
    ui->widget_SAR_MS->hide();
}
void Detection::reset_img_show()
{
    // 设置label的大小
    int h=512,w=512;
    ui->labelImage_opt->resize(w,h);
    ui->labelImage_IR->resize(w,h);
    ui->labelImage_SAR->resize(w,h);
//    // 图像自适应label的大小
//    ui->labelImage->setScaledContents(true);
    ui->labelImage_opt->clear();
    ui->labelImage_IR->clear();
    ui->labelImage_SAR->clear();

}

void Detection::reset_bu_groups()
{
    // 按钮组中不互斥
    this->bu_group_SS.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    QList<QAbstractButton*> list;
    list = this->bu_group_SS.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);
    // 按钮组互斥
    bu_group_SS.setExclusive(true);

    // 按钮组不互斥
    this->bu_group_MS.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    list = this->bu_group_MS.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);

}

void Detection::reset_config_params()
{
    // 初始化算法参数设置
    ui->line_score->setText(QString::number(0.5));
    ui->line_iou_thr->setText(QString::number(0.1));
    ui->line_max_before_nms->setText(QString::number(2000));
}
// 初始化界面
void Detection::init_ui()
{
    // 初始化图像路径界面和图像显示界面
    this->reset_show();

    // 初始化按钮组，全部是未选中的状态
    this->reset_bu_groups();

    // 初始化算法参数设置
    this->reset_config_params();

    // 软件日志初始化
    ui->text_log->clear();
    ui->text_log->setText("请输入图像路径...");

    // 保存路径
    ui->le_savepath->clear();

    ui->stackedWidget_detect->setCurrentWidget(ui->page_SS);
}
