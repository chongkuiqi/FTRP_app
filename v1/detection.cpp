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
        if (this->img_status.opt) widget_ss_show("opt");//ui->widget_opt_SS->show();
        if (this->img_status.IR)  widget_ss_show("IR");//ui->widget_IR_SS->show();
        if (this->img_status.SAR) widget_ss_show("SAR");//ui->widget_SAR_SS->show();
    }
    else
    {
        if (this->img_status.opt) widget_ms_show("opt");//ui->widget_opt_MS->show();
        if (this->img_status.IR)  widget_ms_show("IR");//ui->widget_IR_MS->show();
        if (this->img_status.SAR) widget_ms_show("SAR");//ui->widget_SAR_MS->show();
    }
}
void Detection::widget_ss_show(std::string phase ){
    if(phase=="opt"){
        ui->bu_browse_opt_SS->show();
        ui->label_imgpath_opt_SS->show();
        ui->le_imgpath_opt_SS->show();
    }else if(phase=="IR"){
        ui->bu_browse_IR_SS->show();
        ui->label_imgpath_IR_SS->show();
        ui->le_imgpath_IR_SS->show();
    }else if(phase=="SAR"){
        ui->bu_browse_SAR_SS->show();
        ui->label_imgpath_SAR_SS->show();
        ui->le_imgpath_SAR_SS->show();
    }
}
void Detection::widget_ss_hide(std::string phase ){
    if(phase=="opt"){
        ui->bu_browse_opt_SS->hide();
        ui->label_imgpath_opt_SS->hide();
        ui->le_imgpath_opt_SS->hide();
    }else if(phase=="IR"){
        ui->bu_browse_IR_SS->hide();
        ui->label_imgpath_IR_SS->hide();
        ui->le_imgpath_IR_SS->hide();
    }else if(phase=="SAR"){
        ui->bu_browse_SAR_SS->hide();
        ui->label_imgpath_SAR_SS->hide();
        ui->le_imgpath_SAR_SS->hide();
    }
}
void Detection::widget_ms_show(std::string phase ){
    if(phase=="opt"){
        ui->bu_browse_opt_MS->show();
        ui->label_imgpath_opt_MS->show();
        ui->le_imgpath_opt_MS->show();
    }else if(phase=="IR"){
        ui->bu_browse_IR_MS->show();
        ui->label_imgpath_IR_MS->show();
        ui->le_imgpath_IR_MS->show();
    }else if(phase=="SAR"){
        ui->bu_browse_SAR_MS->show();
        ui->label_imgpath_SAR_MS->show();
        ui->le_imgpath_SAR_MS->show();
    }
}
void Detection::widget_ms_hide(std::string phase ){
    if(phase=="opt"){
        ui->bu_browse_opt_MS->hide();
        ui->label_imgpath_opt_MS->hide();
        ui->le_imgpath_opt_MS->hide();
    }else if(phase=="IR"){
        ui->bu_browse_IR_MS->hide();
        ui->label_imgpath_IR_MS->hide();
        ui->le_imgpath_IR_MS->hide();
    }else if(phase=="SAR"){
        ui->bu_browse_SAR_MS->hide();
        ui->label_imgpath_SAR_MS->hide();
        ui->le_imgpath_SAR_MS->hide();
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
//    std::cout << "cuda is_available:" << torch::cuda::is_available() << std::endl;

    ui->text_log->append("正在获取图像...");

    QString img_path;
    cv::Mat img;
    at::Tensor img_tensor;
    std::vector<c10::IValue> imgs;
    float ratio=1.0;
    int padding_top, padding_left=0;

    // 以下三个if语句，按顺序分别提取可见光、红外、SAR图像，顺序不能乱，否则输入到网络中的图像顺序也是乱的
    if (this->img_status.opt)
    {
        img_path = this->img_paths.opt;
        LoadImage(img_path.toStdString(), img); // CV_8UC3
//        preprocess(img, img_tensor);
        preprocess(img, img_tensor, ratio, padding_top, padding_left);
        imgs.push_back(img_tensor.clone());
    }
    if (this->img_status.IR)
    {
        img_path = this->img_paths.IR;
        LoadImage(img_path.toStdString(), img); // CV_8UC3
//        preprocess(img, img_tensor);
        preprocess(img, img_tensor, ratio, padding_top, padding_left);
        imgs.push_back(img_tensor.clone());
    }
    if (this->img_status.SAR)
    {
        img_path = this->img_paths.SAR;
        LoadImage(img_path.toStdString(), img); // CV_8UC3
//        preprocess(img, img_tensor);
        preprocess(img, img_tensor, ratio, padding_top, padding_left);
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
            model_path = "./model/detection/opt_exp384.pt";
            break;
        // 红外
        case 10:
            model_path = "./model/detection/IR_exp385.pt";
            break;
        // SAR
        case 1:
            model_path = "./model/detection/SAR_exp386.pt";
            break;

        // 可见光+红外
        case 110:
            model_path = "./model/detection/opt_IR_exp21.pt";
            break;
        // 可见光+SAR
        case 101:
            model_path = "./model/detection/opt_SAR_exp22.pt";
            break;
        // 红外+SAR
        case 11:
            model_path = "./model/detection/IR_SAR_exp23.pt";
            break;

        // 可见光+红外+SAR
        case 111:
            model_path = "./model/detection/opt_IR_SAR_exp20.pt";
            break;

        default:
            std::cout << "没有选择任何模型！！！" << std::endl;
            break;
    }


    ui->text_log->append("开始检测...");
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
                  max_before_nms,
                  score_thr, iou_thr, max_before_nms);

    // 边界框转化为原始的图像尺寸
    results.boxes.select(1, 0) = results.boxes.select(1, 0).sub_(padding_left).div_(ratio);
    results.boxes.select(1, 1) = results.boxes.select(1, 1).sub_(padding_top).div_(ratio);
    results.boxes.select(1, 2) = results.boxes.select(1, 2).div_(ratio);
    results.boxes.select(1, 3) = results.boxes.select(1, 3).div_(ratio);
    xywhtheta2points(results.boxes, this->contours);


    QString text = QString("图像：");
    if (this->img_status.opt) text += QString("可见光 ");
    if (this->img_status.IR) text += QString("红外 ");
    if (this->img_status.SAR) text += QString("SAR ");
    text += QString("检测完成！！！");
    ui->text_log->append(text);


    // 显示图像
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
        this->img_results.opt = img.clone();

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
        this->img_results.IR = img.clone();

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
        this->img_results.SAR = img.clone();

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
    // 保存检测结果
    QString path = ui->le_savepath->text();
    QString pathname = ui->le_savepath_name->text();

    QString save_path = path + "/" + pathname;
    QDir dir(save_path);
    if(!dir.exists())
    {
        dir.mkdir(save_path);
    }


    QString img_path;
    //获取图像名称和路径
    QFileInfo imginfo;
    // 图像名称
    QString img_name;
    //文件后缀
    QString fileSuffix;
    // 保存图像
    if (this->img_status.opt)
    {
        img_path = this->img_paths.opt;
        //获取图像名称和路径
        imginfo = QFileInfo(img_path);
        // 图像名称
        img_name = imginfo.fileName();
        //文件后缀
        fileSuffix = imginfo.suffix();

        // 保存图像
        QString save_img_name = img_name;
        save_img_name.replace(QRegExp("."+fileSuffix), QString("_opt."+fileSuffix));
        QString save_img_pathname = save_path + '/' + save_img_name;
        cv::imwrite(save_img_pathname.toStdString(), this->img_results.opt);
    }
    if (this->img_status.IR)
    {
        img_path = this->img_paths.IR;
        //获取图像名称和路径
        imginfo = QFileInfo(img_path);
        // 图像名称
        img_name = imginfo.fileName();
        //文件后缀
        fileSuffix = imginfo.suffix();

        // 保存图像
        QString save_img_name = img_name;
        save_img_name.replace(QRegExp("."+fileSuffix), QString("_IR."+fileSuffix));
        QString save_img_pathname = save_path + '/' + save_img_name;
        cv::imwrite(save_img_pathname.toStdString(), this->img_results.IR);
    }
    if (this->img_status.SAR)
    {
        img_path = this->img_paths.SAR;
        //获取图像名称和路径
        imginfo = QFileInfo(img_path);
        // 图像名称
        img_name = imginfo.fileName();
        //文件后缀
        fileSuffix = imginfo.suffix();

        // 保存图像
        QString save_img_name = img_name;
        save_img_name.replace(QRegExp("."+fileSuffix), QString("_SAR."+fileSuffix));
        QString save_img_pathname = save_path + '/' + save_img_name;
        cv::imwrite(save_img_pathname.toStdString(), this->img_results.SAR);
    }

    // 保存检测结果
    QString txt_name = "results.txt";
    QString txt_save_path = save_path +"/"+ txt_name;

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

    ui->text_log->append("保存完成！！！");

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

//    ui->widget_opt_SS->hide();
//    ui->widget_IR_SS->hide();
//    ui->widget_SAR_SS->hide();
    widget_ss_hide("opt");
    widget_ss_hide("IR");
    widget_ss_hide("SAR");
    widget_ms_hide("opt");
    widget_ms_hide("IR");
    widget_ms_hide("SAR");
//    ui->widget_opt_MS->hide();
//    ui->widget_IR_MS->hide();
//    ui->widget_SAR_MS->hide();
}
void Detection::reset_img_show()
{
    // 设置label的大小
    int h=500,w=500;
    ui->labelImage_opt->resize(w,h);
    ui->labelImage_IR->resize(w,h);
    ui->labelImage_SAR->resize(w,h);
//    // 图像自适应label的大小
//    ui->labelImage->setScaledContents(true);
    ui->labelImage_opt->clear();
    ui->labelImage_IR->clear();
    ui->labelImage_SAR->clear();

    ui->tabWidget_show->setCurrentWidget(ui->opt);

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
    ui->line_score->setText(QString::number(0.2));
    ui->line_iou_thr->setText(QString::number(0.1));
    ui->line_max_before_nms->setText(QString::number(2000));
}
// 初始化界面
void Detection::init_ui()
{
    ui->CB_mode->setCurrentIndex(0);
    // 初始化图像路径界面和图像显示界面
    this->reset_show();

    // 初始化按钮组，全部是未选中的状态
    this->reset_bu_groups();

    // 初始化算法参数设置
    this->reset_config_params();

    // 软件日志初始化
    ui->text_log->clear();
    ui->text_log->setText("请选择检测模式，并输入图像路径...");

    // 保存路径
    ui->le_savepath->clear();

    ui->stackedWidget_detect->setCurrentWidget(ui->page_SS);
}
