//#include "util.h"
#include "probability.h"
#include "ui_probability.h"
#include <QFileDialog>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#include <string>

#include <QDebug>
#include <cmath>
#include "roi_align_rotated_cpu.h"

using std::cout;
using std::endl;

Probability::Probability(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Probability)
{

    // 频谱图像类型初始化
    opt_spectrum.spectrum_type = OPT_SPECTRUM;
    ir_spectrum.spectrum_type = IR_SPECTRUM;
    sar_spectrum.spectrum_type = SAR_SPECTRUM;




    ui->setupUi(this);
    connect(ui->bu_exit, &QPushButton::clicked, this, &QMainWindow::close);

    // 多频谱图像选择按钮，放进一个按钮组
    bu_group_MS.setParent(this);
    bu_group_MS.addButton(ui->CB_opt, 0);
    bu_group_MS.addButton(ui->CB_IR, 1);
    bu_group_MS.addButton(ui->CB_SAR, 2);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_MS, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_group_MS);

    // 读取图像文件路径按钮
    connect(ui->bu_browse_opt, &QPushButton::clicked, this, &Probability::browse_img_opt);
    connect(ui->bu_browse_IR, &QPushButton::clicked, this, &Probability::browse_img_IR);
    connect(ui->bu_browse_SAR, &QPushButton::clicked, this, &Probability::browse_img_SAR);
    connect(ui->bu_browse_opt_2, &QPushButton::clicked, this, &Probability::browse_img_opt_2);
    connect(ui->bu_browse_IR_2, &QPushButton::clicked, this, &Probability::browse_img_IR_2);
    connect(ui->bu_browse_SAR_2, &QPushButton::clicked, this, &Probability::browse_img_SAR_2);



    // 特征提取区域选择按钮组
    bu_group_rois.setParent(this);
    // 按钮组互斥
    bu_group_rois.setExclusive(true);
    bu_group_rois.addButton(ui->RB_fg_bg, 0);
    bu_group_rois.addButton(ui->RB_fg_fg, 1);
    bu_group_rois.addButton(ui->RB_fg_fg2, 2);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_rois, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_group_rois);

    // 读取标签文件路径
    connect(ui->bu_browse_gt_1, &QPushButton::clicked, this, &Probability::browse_gt_1);
    connect(ui->bu_browse_gt_2, &QPushButton::clicked, this, &Probability::browse_gt_2);
    connect(ui->bu_browse_gt_3, &QPushButton::clicked, this, &Probability::browse_gt_3);
    connect(ui->bu_browse_gt_4, &QPushButton::clicked, this, &Probability::browse_gt_4);



    // 目标选择
    connect(ui->CB_roi_choose_1, &QComboBox::currentTextChanged, this, &Probability::deal_roi_choose_1);
    connect(ui->CB_roi_choose_2, &QComboBox::currentTextChanged, this, &Probability::deal_roi_choose_2);
    connect(ui->CB_roi_choose_3, &QComboBox::currentTextChanged, this, &Probability::deal_roi_choose_3);
    connect(ui->CB_roi_choose_4, &QComboBox::currentTextChanged, this, &Probability::deal_roi_choose_4);
    connect(ui->CB_roi_choose_5, &QComboBox::currentTextChanged, this, &Probability::deal_roi_choose_5);



    // 提取区域
    connect(ui->bu_extract_rois, &QPushButton::clicked, this, &Probability::extract_rois);
    // 提取特征
    connect(ui->bu_extract_fe, &QPushButton::clicked, this, &Probability::extract_fe);

    connect(ui->bu_similarity, &QPushButton::clicked, this, &Probability::cal_similarity);

    connect(ui->bu_probability, &QPushButton::clicked, this, &Probability::cal_probability);

    connect(ui->bu_save, &QPushButton::clicked, this, &Probability::save_results);

    // 保存路径
    connect(ui->bu_browse_save, &QPushButton::clicked, this, &Probability::browse_save);


    // 三种图像特征选择按钮，放进一个按钮组
    bu_group_fe.setParent(this);
    // 按钮组中不互斥
    bu_group_fe.setExclusive(false);
    bu_group_fe.addButton(ui->bu_deep_fe, 0);
    bu_group_fe.addButton(ui->bu_gray_fe, 1);
    bu_group_fe.addButton(ui->bu_text_fe, 2);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_fe, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_group_fe);




    // 相似度加权方式选择按钮组
    // 特征加权
    bu_group_fe_weights.setParent(this);
    bu_group_fe_weights.setExclusive(true);
    bu_group_fe_weights.addButton(ui->RB_weights_fe_custom, 0);
    bu_group_fe_weights.addButton(ui->RB_weights_fe_entropy, 1);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_fe_weights, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_group_fe_weights);
    // 频谱加权
    bu_group_sp_weights.setParent(this);
    bu_group_sp_weights.setExclusive(true);
    bu_group_sp_weights.addButton(ui->RB_weights_sp_custom, 0);
    bu_group_sp_weights.addButton(ui->RB_weights_sp_entropy, 1);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_sp_weights, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_group_sp_weights);



    this->init_ui();

}

Probability::~Probability() {delete ui;}


void reset_bu_group(QButtonGroup &button_group, bool set_exclusive)
{
    /*
     * set_exclusive:按钮组互斥或非互斥状态，true表示互斥
    */
    // 按钮组中不互斥
    button_group.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    QList<QAbstractButton*> list;
    list = button_group.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);
    // 按钮组互斥或非互斥
    button_group.setExclusive(set_exclusive);
}

void Probability::reset_imgpath_show()
{

    ui->le_imgpath_opt->clear();
    ui->le_imgpath_IR->clear();
    ui->le_imgpath_SAR->clear();


//    ui->widget_opt->hide();
    widget_opt_hide();
    widget_IR_hide();
    widget_SAR_hide();
//    ui->widget_IR->hide();
//    ui->widget_SAR->hide();
}
void Probability::widget_opt_hide(){
    ui->bu_browse_opt->hide();
    ui->le_imgpath_opt->hide();
    ui->label_imgpath_opt->hide();
}
void Probability::widget_opt_show(){
    ui->bu_browse_opt->show();
    ui->le_imgpath_opt->show();
    ui->label_imgpath_opt->show();
}
void Probability::widget_IR_hide(){
    ui->bu_browse_IR->hide();
    ui->le_imgpath_IR->hide();
    ui->label_imgpath_IR->hide();
}
void Probability::widget_IR_show(){
    ui->bu_browse_IR->show();
    ui->le_imgpath_IR->show();
    ui->label_imgpath_IR->show();
}
void Probability::widget_SAR_hide(){
    ui->bu_browse_SAR->hide();
    ui->le_imgpath_SAR->hide();
    ui->label_imgpath_SAR->hide();
}
void Probability::widget_SAR_show(){
    ui->bu_browse_SAR->show();
    ui->le_imgpath_SAR->show();
    ui->label_imgpath_SAR->show();
}
void Probability::reset_img_show()
{
    // 设置label的大小
//    int h=400,w=400;
    int h=300,w=300;
    ui->labelImage_opt->resize(w,h);
    ui->labelImage_IR->resize(w,h);
    ui->labelImage_SAR->resize(w,h);
//    // 图像自适应label的大小
//    ui->labelImage->setScaledContents(true);
    ui->labelImage_opt->clear();
    ui->labelImage_IR->clear();
    ui->labelImage_SAR->clear();

    ui->labelImage_opt_2->resize(w,h);
    ui->labelImage_IR_2->resize(w,h);
    ui->labelImage_SAR_2->resize(w,h);
//    // 图像自适应label的大小
//    ui->labelImage->setScaledContents(true);
    ui->labelImage_opt_2->clear();
    ui->labelImage_IR_2->clear();
    ui->labelImage_SAR_2->clear();

    ui->tabWidget_show->setCurrentWidget(ui->opt);


}
void Probability::on_bu_group_MS(QAbstractButton *button)
{

    this->img_status = {false, false, false};
    // 先reset区域，因为这里也有显示图像，后面reset_img_show再把图像清空
    this->reset_rois();

    // 初始化图像路径界面和图像显示界面
    this->reset_imgpath_show();
    this->reset_img_show();

    // 频谱加权界面也reset
    this->reset_weights_sp();


    // 遍历按钮，获取选中状态
    QList<QAbstractButton*> list = this->bu_group_MS.buttons();
    foreach (QAbstractButton *pCheckBox, list)
    {
        // 获取按钮名称和状态
       QString bu_name = pCheckBox->text();
       bool status = pCheckBox->isChecked();

       if (bu_name == QString("可见光")) this->img_status.opt = status;
       if (bu_name == QString("红外")) this->img_status.IR = status;
       if (bu_name == QString("SAR")) this->img_status.SAR = status;

    }

    if (this->img_status.opt) widget_opt_show();//ui->widget_opt->show();
    if (this->img_status.IR)  widget_IR_show();//ui->widget_IR->show();
    if (this->img_status.SAR) widget_SAR_show();//ui->widget_SAR->show();

}

void Probability::reset_MS()
{
    this->img_status = {false, false, false};
    // 初始化图像路径界面和图像显示界面
    this->reset_imgpath_show();
    this->reset_img_show();

    // 初始化按钮组
    reset_bu_group(this->bu_group_MS, false);

    // 频谱加权界面设置为空白
    this->reset_weights_sp();

}



void Probability::browse_img(SpectrumType img_type, bool is_img_1)
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

    if (img_type==OPT_SPECTRUM)
    {
        if (is_img_1)
        {
            // 加载图像
            LoadImage(img_path.toStdString(), this->opt_spectrum.img_1); // CV_8UC3
            this->opt_spectrum.img_1_path = img_path.toStdString();
            ui->le_imgpath_opt->setText(img_path);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg->scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
            ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));
        }
        else
        {
            // 加载图像
            LoadImage(img_path.toStdString(), this->opt_spectrum.img_2); // CV_8UC3
            this->opt_spectrum.img_2_path = img_path.toStdString();
            ui->le_imgpath_opt_2->setText(img_path);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg->scaled(ui->labelImage_opt_2->size(),Qt::KeepAspectRatio);
            ui->labelImage_opt_2->setPixmap(QPixmap::fromImage(dest));
            if (this->opt_spectrum.img_2_path == this->opt_spectrum.img_1_path) {ui->text_log->append("警告：可见光图像1和可见光图像2相同！！！");}

        }

    }

    if (img_type==IR_SPECTRUM)
    {
        if (is_img_1)
        {
            // 加载图像
            LoadImage(img_path.toStdString(), this->ir_spectrum.img_1); // CV_8UC3
            this->ir_spectrum.img_1_path = img_path.toStdString();
            ui->le_imgpath_IR->setText(img_path);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg->scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
            ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));
        }
        else
        {
            // 加载图像
            LoadImage(img_path.toStdString(), this->ir_spectrum.img_2); // CV_8UC3
            this->ir_spectrum.img_2_path = img_path.toStdString();
            ui->le_imgpath_IR_2->setText(img_path);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg->scaled(ui->labelImage_IR_2->size(),Qt::KeepAspectRatio);
            ui->labelImage_IR_2->setPixmap(QPixmap::fromImage(dest));
            if (this->ir_spectrum.img_2_path == this->ir_spectrum.img_1_path) {ui->text_log->append("警告：红外图像1和红外图像2相同！！！");}

        }

    }

    if (img_type==SAR_SPECTRUM)
    {
        if (is_img_1)
        {
            // 加载图像
            LoadImage(img_path.toStdString(), this->sar_spectrum.img_1); // CV_8UC3
            this->sar_spectrum.img_1_path = img_path.toStdString();
            ui->le_imgpath_SAR->setText(img_path);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg->scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
            ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
        }
        else
        {
            // 加载图像
            LoadImage(img_path.toStdString(), this->sar_spectrum.img_2); // CV_8UC3
            this->sar_spectrum.img_2_path = img_path.toStdString();
            ui->le_imgpath_SAR_2->setText(img_path);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg->scaled(ui->labelImage_SAR_2->size(),Qt::KeepAspectRatio);
            ui->labelImage_SAR_2->setPixmap(QPixmap::fromImage(dest));
            if (this->sar_spectrum.img_2_path == this->sar_spectrum.img_1_path) {ui->text_log->append("警告：SAR图像1和SAR图像2相同！！！");}

        }

    }

}

void Probability::browse_img_opt() {browse_img(OPT_SPECTRUM, true);}
void Probability::browse_img_IR()  {browse_img(IR_SPECTRUM, true);}
void Probability::browse_img_SAR() {browse_img(SAR_SPECTRUM, true);}
void Probability::browse_img_opt_2() {browse_img(OPT_SPECTRUM, false);}
void Probability::browse_img_IR_2()  {browse_img(IR_SPECTRUM, false);}
void Probability::browse_img_SAR_2() {browse_img(SAR_SPECTRUM, false);}




// 提取特征区域选择
void Probability::on_bu_group_rois(QAbstractButton *button)
{
    QString bu_text = button->text();
    if (bu_text==QString("目标-背景")) this->rois_type = FG_BG;
    else if (bu_text==QString("目标-目标（单张图像）")) this->rois_type = FG_FG;
    else if (bu_text==QString("目标-目标（两张图像）")) this->rois_type = FG_FG2;
    else std::cout << "未选择特征提取区域" << std::endl;

    this->reset_rois_show();

    if (this->rois_type == FG_BG) ui->SW_choose_rois->setCurrentWidget(ui->page_fg_bg);
    if (this->rois_type == FG_FG) ui->SW_choose_rois->setCurrentWidget(ui->page_fg_fg);

    if (this->rois_type == FG_BG || this->rois_type == FG_FG)
    {
        if(this->img_status.opt)
        {
            this->opt_spectrum.img_2 = this->opt_spectrum.img_1.clone();
            this->opt_spectrum.img_2_path = this->opt_spectrum.img_1_path;
        }
        if(this->img_status.IR)
        {
            this->ir_spectrum.img_2 = this->ir_spectrum.img_1.clone();
            this->ir_spectrum.img_2_path = this->ir_spectrum.img_1_path;
        }
        if(this->img_status.SAR)
        {
            this->sar_spectrum.img_2 = this->sar_spectrum.img_1.clone();
            this->sar_spectrum.img_2_path = this->sar_spectrum.img_1_path;
        }
    };


    if (this->rois_type == FG_FG2)
    {
        ui->SW_choose_rois->setCurrentWidget(ui->page_fg_fg2);

        if (this->img_status.opt) ui->widget_opt_2->show();
        if (this->img_status.IR) ui->widget_IR_2->show();
        if (this->img_status.SAR) ui->widget_SAR_2->show();
    };


}

void reset_QCB(QComboBox *CB_box)
{
    // 如果QComboBox只有一个-1，此时clear会导致text发生变化，从而导致信号触发，因此在clear前先断开信号连接
    CB_box->blockSignals(true);//true屏蔽信号
    CB_box->clear();
    CB_box->addItem(QString(" "));
    CB_box->blockSignals(false);//false取消屏蔽信号
}
void Probability::reset_rois_show()
{

    ui->le_gtpath_1->clear();
    ui->le_gtpath_2->clear();
    ui->le_gtpath_3->clear();
    ui->le_gtpath_4->clear();

    ui->le_bg_ratio->setText(QString::number(2));

    // 如果QComboBox只有一个-1，此时clear会导致text发生变化，从而导致信号触发，因此在clear前先断开信号连接
    reset_QCB(ui->CB_roi_choose_1);
    reset_QCB(ui->CB_roi_choose_2);
    reset_QCB(ui->CB_roi_choose_3);
    reset_QCB(ui->CB_roi_choose_4);
    reset_QCB(ui->CB_roi_choose_5);


    // 显示第一张图像
    if (this->img_status.opt)
    {
        cv::Mat img;
        img = this->opt_spectrum.img_1.clone();
        //显示图像
        QImage srcimg = MatToImage(img);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.IR)
    {
        cv::Mat img;
        img = this->ir_spectrum.img_1.clone();
        //显示图像
        QImage srcimg = MatToImage(img);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.SAR)
    {
        cv::Mat img;
        img = this->sar_spectrum.img_1.clone();
        //显示图像
        QImage srcimg = MatToImage(img);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
    }



    // 设置label的大小
    int h=300,w=300;

    ui->labelImage_opt_2->resize(w,h);
    ui->labelImage_IR_2->resize(w,h);
    ui->labelImage_SAR_2->resize(w,h);
//    // 图像自适应label的大小
//    ui->labelImage->setScaledContents(true);
    ui->labelImage_opt_2->clear();
    ui->labelImage_IR_2->clear();
    ui->labelImage_SAR_2->clear();


    // 第2张图像相关的图像路径等进行隐藏
    ui->le_imgpath_opt_2->clear();
    ui->le_imgpath_IR_2->clear();
    ui->le_imgpath_SAR_2->clear();

    ui->widget_opt_2->hide();
    ui->widget_IR_2->hide();
    ui->widget_SAR_2->hide();

    ui->tabWidget_show_2->setCurrentWidget(ui->opt_2);

}

void Probability::reset_rois()
{

    reset_bu_group(this->bu_group_rois, true);

    // 区域选择置于空白页面
    ui->SW_choose_rois->setCurrentWidget(ui->page_blank);
    this->reset_rois_show();
}

void Probability::browse_gt_1()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Text files(*.txt)"
                );

    ui->le_gtpath_1->setText(path);

    // 把所有框都画出来，供给用户选择
    LoadBoxes(path, this->contours_1);

    // 可选的框展示给用户
    ui->CB_roi_choose_1->clear();
    int num_boxes = this->contours_1.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_1->addItem(QString::number(i));
    }
    ui->CB_roi_choose_1->addItem(QString(" "));
    ui->CB_roi_choose_1->setCurrentIndex(num_boxes); //显示-1
}
void Probability::browse_gt_2()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Text files(*.txt)"
                );

    ui->le_gtpath_2->setText(path);

    // 把所有框都画出来，供给用户选择
    LoadBoxes(path, this->contours_1);


    // 可选的框展示给用户
    ui->CB_roi_choose_2->clear();
    ui->CB_roi_choose_3->clear();
    int num_boxes = this->contours_1.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_2->addItem(QString::number(i));
        ui->CB_roi_choose_3->addItem(QString::number(i));
    }
    ui->CB_roi_choose_2->addItem(QString(" "));
    ui->CB_roi_choose_2->setCurrentIndex(num_boxes); //显示-1
    ui->CB_roi_choose_3->addItem(QString(" "));
    ui->CB_roi_choose_3->setCurrentIndex(num_boxes); //显示-1
}
void Probability::browse_gt_3()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Text files(*.txt)"
                );

    ui->le_gtpath_3->setText(path);

    // 把所有框都画出来，供给用户选择
    LoadBoxes(path, this->contours_1);

    // 可选的框展示给用户
    ui->CB_roi_choose_4->clear();
    int num_boxes = this->contours_1.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_4->addItem(QString::number(i));
    }
    ui->CB_roi_choose_4->addItem(QString(" "));
    ui->CB_roi_choose_4->setCurrentIndex(num_boxes); //显示-1
}
void Probability::browse_gt_4()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Text files(*.txt)"
                );

    ui->le_gtpath_4->setText(path);


    // 把所有框都画出来，供给用户选择
    LoadBoxes(path, this->contours_2);

    // 可选的框展示给用户
    ui->CB_roi_choose_5->clear();
    int num_boxes = this->contours_2.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_5->addItem(QString::number(i));
    }
    ui->CB_roi_choose_5->addItem(QString(" "));
    ui->CB_roi_choose_5->setCurrentIndex(num_boxes); //显示-1

}


void Probability::deal_roi_choose_1(const QString &text)
{
    std::vector<std::vector<cv::Point>> contour_1;
    std::vector<std::vector<cv::Point>> contour_2;

    if (text != QString(" "))
    {
        int box_id = text.toInt();
        contour_1.push_back( this->contours_1[box_id] );
    }
    this->show_all_objs_img_1(contour_1, contour_2);
}
void Probability::deal_roi_choose_2(const QString &text)
{
    std::vector<std::vector<cv::Point>> contour_1;
    std::vector<std::vector<cv::Point>> contour_2;

    if (text != QString(" "))
    {
        int box_id = text.toInt();
        contour_1.push_back( this->contours_1[box_id] );
    }

    QString text_2 = ui->CB_roi_choose_3->currentText();
    if (text_2 != QString(" "))
    {
        int box_id = text_2.toInt();
        contour_2.push_back( this->contours_1[box_id] );
    }


    this->show_all_objs_img_1(contour_1, contour_2);

}

void Probability::deal_roi_choose_3(const QString &text)
{
    std::vector<std::vector<cv::Point>> contour_1;
    std::vector<std::vector<cv::Point>> contour_2;

    QString text_1 = ui->CB_roi_choose_2->currentText();
    if (text_1 != QString(" "))
    {
        int box_id = text_1.toInt();
        contour_1.push_back( this->contours_1[box_id] );
    }

    if (text != QString(" "))
    {
        int box_id = text.toInt();
        contour_2.push_back( this->contours_1[box_id] );
    }
    this->show_all_objs_img_1(contour_1, contour_2);

}

void Probability::deal_roi_choose_4(const QString &text) {this->deal_roi_choose_1(text);}


void Probability::deal_roi_choose_5(const QString &text)
{
    std::vector<std::vector<cv::Point>> contour_2;

    if (text != QString(" "))
    {
        int box_id = text.toInt();
        contour_2.push_back( this->contours_2[box_id] );
    }
    this->show_all_objs_img_2(contour_2);

}
void Probability::show_all_objs_img_1(std::vector<std::vector<cv::Point>> contour_1, std::vector<std::vector<cv::Point>> contour_2)
{

    cv::Scalar color_1 = cv::Scalar(0,255,0), color_2 = cv::Scalar(255,0,0);

    if (this->img_status.opt)
    {
        cv::Mat img_result;
        // 画出所有框
        draw_rboxes_ids(this->opt_spectrum.img_1, img_result, this->contours_1);
        // 画出两个预选框
        if(contour_1.size() > 0) draw_rboxes(img_result, img_result, contour_1, -1, color_1);
        if(contour_2.size() > 0) draw_rboxes(img_result, img_result, contour_2, -1, color_2);

        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.IR)
    {
        cv::Mat img_result;
        // 画出所有框
        draw_rboxes_ids(this->ir_spectrum.img_1, img_result, this->contours_1);
        // 画出两个预选框
        if(contour_1.size() > 0) draw_rboxes(img_result, img_result, contour_1, -1, color_1);
        if(contour_2.size() > 0) draw_rboxes(img_result, img_result, contour_2, -1, color_2);
        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.SAR)
    {
        cv::Mat img_result;
        // 画出所有框
        draw_rboxes_ids(this->sar_spectrum.img_1, img_result, this->contours_1);
        // 画出两个预选框
        if(contour_1.size() > 0) draw_rboxes(img_result, img_result, contour_1, -1, color_1);
        if(contour_2.size() > 0) draw_rboxes(img_result, img_result, contour_2, -1, color_2);
        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
    }


}

void Probability::show_all_objs_img_2(std::vector<std::vector<cv::Point>> contour)
{
    cv::Scalar color_2 = cv::Scalar(255,0,0);

    if (this->img_status.opt)
    {
        cv::Mat img_result;
        // 画出所有框
        draw_rboxes_ids(this->opt_spectrum.img_2, img_result, this->contours_2);
        // 画出预选框
        if(contour.size() > 0) draw_rboxes(img_result, img_result, contour, -1, color_2);

        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_opt_2->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt_2->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.IR)
    {
        cv::Mat img_result;
        // 画出所有框
        draw_rboxes_ids(this->ir_spectrum.img_2, img_result, this->contours_2);
        // 画出预选框
        if(contour.size() > 0) draw_rboxes(img_result, img_result, contour, -1, color_2);

        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_IR_2->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR_2->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.SAR)
    {
        cv::Mat img_result;
        // 画出所有框
        draw_rboxes_ids(this->sar_spectrum.img_2, img_result, this->contours_2);
        // 画出预选框
        if(contour.size() > 0) draw_rboxes(img_result, img_result, contour, -1, color_2);

        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_SAR_2->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR_2->setPixmap(QPixmap::fromImage(dest));
    }
}

void Probability::show_rois_img_1(std::vector<std::vector<cv::Point>> contour_1, std::vector<std::vector<cv::Point>> contour_2)
{

    cv::Scalar color_1 = cv::Scalar(0,255,0), color_2 = cv::Scalar(255,0,0);

    if (this->img_status.opt)
    {
        cv::Mat img_result;

        // 画出选中的roi
        draw_rboxes(this->opt_spectrum.img_1, img_result, contour_1, -1, color_1);
        if(contour_2.size() > 0) draw_rboxes(img_result, img_result, contour_2, -1, color_2);

        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_opt->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.IR)
    {
        cv::Mat img_result;
        // 画出选中的roi
        draw_rboxes(this->ir_spectrum.img_1, img_result, contour_1, -1, color_1);
        if(contour_2.size() > 0) draw_rboxes(img_result, img_result, contour_2, -1, color_2);
        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_IR->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.SAR)
    {
        cv::Mat img_result;
        // 画出选中的roi
        draw_rboxes(this->sar_spectrum.img_1, img_result, contour_1, -1, color_1);
        if(contour_2.size() > 0) draw_rboxes(img_result, img_result, contour_2, -1, color_2);
        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
    }

}

void Probability::show_rois_img_2(std::vector<std::vector<cv::Point>> contour)
{
    cv::Scalar color_1 = cv::Scalar(0,255,0), color_2 = cv::Scalar(255,0,0);

    if (this->img_status.opt)
    {
        cv::Mat img_result;

        // 画出选中的roi
        draw_rboxes(this->opt_spectrum.img_2, img_result, contour, -1, color_2);
        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_opt_2->size(),Qt::KeepAspectRatio);
        ui->labelImage_opt_2->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.IR)
    {
        cv::Mat img_result;
        // 画出选中的roi
        draw_rboxes(this->ir_spectrum.img_2, img_result, contour, -1, color_2);
        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_IR_2->size(),Qt::KeepAspectRatio);
        ui->labelImage_IR_2->setPixmap(QPixmap::fromImage(dest));
    }
    if (this->img_status.SAR)
    {
        cv::Mat img_result;
        // 画出选中的roi
        draw_rboxes(this->sar_spectrum.img_2, img_result, contour, -1, color_2);
        //显示图像
        QImage srcimg = MatToImage(img_result);
        // 图像缩放到label的大小，并保持长宽比
        QImage dest = srcimg.scaled(ui->labelImage_SAR_2->size(),Qt::KeepAspectRatio);
        ui->labelImage_SAR_2->setPixmap(QPixmap::fromImage(dest));
    }
}

void Probability::extract_rois()
{
    int box_id_1, box_id_2;
    std::vector<std::vector<cv::Point>> contour_1, contour_2;
    RBox rbox_1, rbox_2;
    if (this->rois_type == FG_BG)
    {

        // 确定第一个框
        box_id_1 = ui->CB_roi_choose_1->currentText().toInt();
        std::vector<cv::Point> contour = this->contours_1[box_id_1];
        // 存储选中的边界框
        this->contour_1.clear();
        this->contour_1.assign(contour.begin(), contour.end());
        // 将选中的框画在图像上
        contour_1.clear();
        contour_1.push_back(contour);


        // 确定第二个框
        // 选择背景区域
        float bg_ratio = ui->le_bg_ratio->text().toFloat();

//        cv::RotatedRect fg_rbox;
        RBox fg_rbox;
        points2rbox(this->contour_1, fg_rbox);
        this->rbox_1 = fg_rbox;


        cv::Point2f center = fg_rbox.center;
        // 单位是角度
        float angle = fg_rbox.angle;
        float w = fg_rbox.size.width;
        float h = fg_rbox.size.height;

        // 背景区域
//        bg_ratio = (-(w+h)+sqrt(pow(w,2) + 6*w*h + pow(h,2))) / 2.0 / h;
        float padding = bg_ratio * h / 2;
        float bg_w = w+2*padding;
        float bg_h = h+2*padding;
//        float bg_w = bg_ratio*w;
//        float bg_h = bg_ratio*h;

//        cv::RotatedRect bg_rbox = cv::RotatedRect(center, cv::Size2f(bg_w,bg_h), angle);
//        std::cout << "w:" << w << ",h:"<< h << ", r:" << bg_ratio << std::endl;

        RBox bg_rbox = {center, cv::Size2f(bg_w,bg_h), angle};
        this->rbox_2 = bg_rbox;


        this->contour_2.clear();
        rbox2points(this->contour_2, bg_rbox);
        contour_2.clear();
        contour_2.push_back(this->contour_2);


        // 画出选中的roi
        this->show_rois_img_1(contour_1, contour_2);

    }

    else if (this->rois_type == FG_FG)
    {
        // 确定第一个框
        box_id_1 = ui->CB_roi_choose_2->currentText().toInt();
        std::vector<cv::Point> contour = this->contours_1[box_id_1];
        // 存储选中的边界框
        this->contour_1.clear();
        this->contour_1.assign(contour.begin(), contour.end());
        // 将选中的框画在图像上
        contour_1.clear();
        contour_1.push_back(contour);


        // 确定第二个框
        box_id_2 = ui->CB_roi_choose_3->currentText().toInt();
        std::vector<cv::Point> contour2 = this->contours_1[box_id_2];
        // 存储选中的边界框
        this->contour_2.clear();
        this->contour_2.assign(contour2.begin(), contour2.end());
        // 将选中的框画在图像上
        contour_2.clear();
        contour_2.push_back(contour2);


        points2rbox(this->contour_1, this->rbox_1);
        points2rbox(this->contour_2, this->rbox_2);

        // 画出选中的roi
        this->show_rois_img_1(contour_1, contour_2);

    }

    else if (this->rois_type == FG_FG2)
    {
        // 确定第一个框
        box_id_1 = ui->CB_roi_choose_4->currentText().toInt();
        std::vector<cv::Point> contour = this->contours_1[box_id_1];
        // 存储选中的边界框
        this->contour_1.clear();
        this->contour_1.assign(contour.begin(), contour.end());
        // 将选中的框画在图像上
        contour_1.clear();
        contour_1.push_back(contour);

        contour_2.clear();
        // 画出选中的roi
        this->show_rois_img_1(contour_1, contour_2);


        // 确定第二个框
        box_id_2 = ui->CB_roi_choose_5->currentText().toInt();
        std::vector<cv::Point> contour2 = this->contours_2[box_id_2];
        // 存储选中的边界框
        this->contour_2.clear();
        this->contour_2.assign(contour2.begin(), contour2.end());
        // 将选中的框画在图像上
        contour_2.clear();
        contour_2.push_back(contour2);


        points2rbox(this->contour_1, this->rbox_1);
        points2rbox(this->contour_2, this->rbox_2);


        // 画出选中的roi
        this->show_rois_img_2(contour_2);

    }

}



// 特征选择和提取
void Probability::on_bu_group_fe(QAbstractButton *button)
{
    this->fe_status = {false, false, false};
    // 遍历按钮，获取选中状态
    QList<QAbstractButton*> list = this->bu_group_fe.buttons();
    foreach (QAbstractButton *pCheckBox, list)
    {
        // 获取按钮名称和状态
       QString bu_name = pCheckBox->text();
       bool status = pCheckBox->isChecked();

       if (bu_name == QString("深度学习特征")) this->fe_status.deep = status;
       if (bu_name == QString("灰度特征")) this->fe_status.gray = status;
       if (bu_name == QString("纹理特征")) this->fe_status.text = status;
    }

    // 特征加权界面设置为空白
    this->reset_weights_fe();
}

void Probability::reset_fe()
{
    reset_bu_group(this->bu_group_fe, false);

    // 特征加权界面设置为空白
    this->reset_weights_fe();

}

int map_roi_levels(float roi_scale, int num_levels, int finest_scale=56)
{

    int level_id = floor(log2(roi_scale / finest_scale + 1e-6));
    level_id = (level_id<0) ? 0 : level_id;
    level_id = (level_id>(num_levels-1)) ? (num_levels-1) : level_id;

    return level_id;
}

void get_fe_deep(cv::Mat &img, const at::Tensor &rbox, at::Tensor &roi_fe_deep, SpectrumType spectrum_type)
{
    std::string model_path;
    switch(spectrum_type)
    {
        case OPT_SPECTRUM:
            model_path = "./model/extract_fe/opt_exp384_fe_script.pt";
            break;
        case IR_SPECTRUM:
            model_path = "./model/extract_fe/IR_exp385_fe_script.pt";
            break;
        case SAR_SPECTRUM:
            model_path = "./model/extract_fe/SAR_exp386_fe_script.pt";
            break;
        default:
            std::cout << "没有选择合适的模型" << std::endl;
            break;
    }


    torch::Device device(torch::kCPU);

    // 提取第一个区域的特征
    at::Tensor input_tensor;
    preprocess(img, input_tensor);
    input_tensor = input_tensor.to(device);
//     cout << "input img size:" << input_tensor.sizes() << endl;

    torch::NoGradGuard no_grad;
    torch::jit::script::Module model;
    model = torch::jit::load(model_path, device);

    c10::intrusive_ptr<at::ivalue::Tuple> output;
    output = model.forward({input_tensor}).toTuple();



    int out_h=7, out_w=7, sample_num=2;
    int finest_scale=56; //Scale threshold of mapping to level 0.
    int featmap_strides[3] = {8,16,32};

    // 获得roi所在的特征层级
    int num_levels=3;
    float roi_1_scale = (rbox[0][3] * rbox[0][4]).item().toFloat();
    int feat_level_id = map_roi_levels(roi_1_scale, num_levels, finest_scale);
    at::Tensor feat = output->elements()[feat_level_id].toTensor();

    // 这里必须是浮点数除法！！！，否则spatial_scale=0
    float spatial_scale = 1./featmap_strides[feat_level_id];
    int chn = feat.sizes()[1];
    int num_rboxes = rbox.sizes()[0];

//    feat_level_id = 0;
    roi_fe_deep = feat.new_zeros({num_rboxes, chn, out_h, out_w});
//    std::cout << feat << std::endl;
    roi_align_rotated_forward_cpu(feat, rbox, out_h, out_w, spatial_scale, sample_num, roi_fe_deep);
    roi_fe_deep = roi_fe_deep.reshape(-1);

}

void extract_fe_deep(SingleSpectrum &specturm_img, const RBox &rbox_1, const RBox &rbox_2, bool is_fg_bg)
{

    // 提取第一个区域的特征
    cv::Mat img_1 = specturm_img.img_1.clone();
    at::Tensor rois_1;

    rbox2xywhtheta(rbox_1, rois_1);
    // 只有一张图像
    float batch_id =0.0;
    rois_1 = torch::cat({at::tensor({batch_id}), rois_1}, 0);
    // shape [N, 6(batch_id, x(像素单位), y, w, h, theta(弧度))]
    rois_1 = rois_1.unsqueeze(0);


//    at::Tensor roi
//      rois = at::Tensor(
//                {
//                    {0.0, 427.9353,616.8455, 119.1755,14.5517, -0.3343},
//                    {0.0, 60.4593, 156.7023, 186.1304, 22.0563, 1.5757}
//                }
//                ).to(device);

    at::Tensor roi_1_fe;
    get_fe_deep(img_1, rois_1, roi_1_fe, specturm_img.spectrum_type);


    // 提取第二个区域的特征
    cv::Mat img_2 = specturm_img.img_2.clone();
    at::Tensor rois_2;
    rbox2xywhtheta(rbox_2, rois_2);
//    rois_2[4] = -0.3122;
    // 只有一张图像
    rois_2 = torch::cat({at::tensor({batch_id}), rois_2}, 0);
    // shape [N, 6(batch_id, x(像素单位), y, w, h, theta(弧度))]
    rois_2 = rois_2.unsqueeze(0);

    at::Tensor roi_2_fe;
    get_fe_deep(img_2, rois_2, roi_2_fe, specturm_img.spectrum_type);

    if (is_fg_bg) roi_2_fe = roi_2_fe - roi_1_fe;

    specturm_img.roi_fe_1.deep = roi_1_fe.clone();
    specturm_img.roi_fe_2.deep = roi_2_fe.clone();

}

// 将旋转框长边旋转到水平方向，并根据图像边界进行截断
cv::Rect get_hbox(RBox rbox, int img_h, int img_w)
{
    cv::Point2f center = rbox.center;

    int x = (int)(center.x);
    int y = (int)(center.y);
    int w = (int)(rbox.size.width);
    int h = (int)(rbox.size.height);
    // 左上、右下角点的坐标,只有长边旋转到水平方向，才能这样计算
    int x1 = x - w/2;
    int y1 = y - h/2;
    int x2 = x + w/2;
    int y2 = y + h/2;
    // 根据图像边界，对左上右下角点坐标进行截断

    x1 = (x1<0) ? 0 : x1;
    x1 = (x1>img_w) ? img_w : x1;
    y1 = (y1<0) ? 0 : y1;
    y1 = (y1>img_h) ? img_h : y1;
    x2 = (x2<0) ? 0 : x2;
    x2 = (x2>img_w) ? img_w : x2;
    y2 = (y2<0) ? 0 : y2;
    y2 = (y2>img_h) ? img_h : y2;
    // 左上角点坐标，w,h
    cv::Rect hbox(x1,y1,x2-x1,y2-y1);

    return hbox;
}

void get_hist(cv::Mat & img, cv::Mat & hist)
{
    // 先把图像转化为单个通道
    if (img.channels() >1) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    //定义参数变量
//    // 区间个数
//    int bins = 64;
//    int histSize[] = {bins, bins};
//    // 区间范围
//    float bin_range[] = { 0, 255 };
//    const float* ranges[] = { bin_range, bin_range };
//    // 提取的图像通道数
//    int channels[] = {0, 1};
    // 区间个数
    int bins = 64;
    int *histSize = &bins;
    // 区间范围
    float bin_range[2] = { 0, 255 };
    const float* ranges[] = { bin_range };
    // 提取的图像通道数
    int *channels = 0;

//    void calcHist( const Mat* images, int nimages,
//                              const int* channels, InputArray mask,
//                              OutputArray hist, int dims, const int* histSize,
//                              const float** ranges, bool uniform = true, bool accumulate = false );

    // 计算得到直方图数据
    cv::calcHist( &img, 1, channels, cv::Mat(), // do not use mask
        hist, 1, histSize, ranges);
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

void get_fe_gray(cv::Mat &img, const RBox &rbox, cv::Mat &hist)
{

    int img_h=img.rows, img_w=img.cols;
    cv::Point2f center = rbox.center;
    // 单位是角度
    float angle = rbox.angle;
    cv::Rect rect_roi = get_hbox(rbox, img_h, img_w);

    // 旋转图像
    cv::Mat img_rotate;
    // S2ANet的angle是顺时针从水平转到长边的角度。getRotationMatrix2D函数的angle参数为正表示逆时针旋转。
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, img_rotate, M, cv::Size(img_w, img_h), cv::INTER_LINEAR, 0);
    cv::Mat img_roi = img_rotate(rect_roi);

//    std::cout << "angle:" << angle <<std::endl;
//    cv::rectangle(img_rotate, rect_roi, cv::Scalar(255,0,0));
//    cv::imshow("images", img_rotate);
//    cv::waitKey(0);

    get_hist(img_roi, hist);

}
void extract_fe_gray(SingleSpectrum &specturm_img, const RBox &rbox_1, const RBox &rbox_2, bool is_fg_bg)
{

//    std::cout << "box1:" << this->contour_1 << std::endl;
//    std::cout << "box2:" << this->contour_2 << std::endl;

    // 提取第一个区域的特征
    cv::Mat img_1 = specturm_img.img_1.clone();
    cv::Mat roi_hist_1;
    get_fe_gray(img_1, rbox_1, roi_hist_1);


    // 提取第二个区域的特征
    cv::Mat img_2 = specturm_img.img_2.clone();
    cv::Mat roi_hist_2;
    get_fe_gray(img_2, rbox_2, roi_hist_2);
//    roi_hist_2 = roi_hist_2 - roi_hist_1;

//    double roimaxVal=0, bgmaxVal=0;
//    minMaxLoc(roi_hist, 0, &roimaxVal, 0, 0);
//    minMaxLoc(bg_hist, 0, &bgmaxVal, 0, 0);

//    roi_hist = roi_hist / roimaxVal;
//    bg_hist = bg_hist / bgmaxVal;
    at::Tensor roi_1_hist_tensor = torch::from_blob(roi_hist_1.data, {roi_hist_1.rows, roi_hist_1.cols, roi_hist_1.channels()}, torch::kFloat);
    at::Tensor roi_2_hist_tensor = torch::from_blob(roi_hist_2.data, {roi_hist_2.rows, roi_hist_2.cols, roi_hist_2.channels()}, torch::kFloat);

    roi_1_hist_tensor = roi_1_hist_tensor.reshape(-1);
    roi_2_hist_tensor = roi_2_hist_tensor.reshape(-1);

    // 背景区域减去前景区域
    if (is_fg_bg) roi_2_hist_tensor = roi_2_hist_tensor - roi_1_hist_tensor;


    roi_1_hist_tensor /= roi_1_hist_tensor.sum();
    roi_2_hist_tensor /= roi_2_hist_tensor.sum();


    specturm_img.roi_fe_1.gray = roi_1_hist_tensor;
    specturm_img.roi_fe_2.gray = roi_2_hist_tensor;


    // 这里直接把特征相似度计算出来，因为灰度特征的相似度总是和纹理特征相似度有所冲突
    float sim = 0.0;
    sim = torch::sqrt(
                torch::mul(specturm_img.roi_fe_1.gray, specturm_img.roi_fe_2.gray)
                ).sum().item().toFloat();

    specturm_img.fe_similarity.gray = sim;
}


void get_fe_text(cv::Mat &img, const RBox &rbox, cv::Mat &roi_fe_text)
{
    // 先把图像转化为单个通道
    if (img.channels() >1) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    int img_h=img.rows, img_w=img.cols;
    cv::Point2f center = rbox.center;
    // 单位是角度
    float angle = rbox.angle;
    cv::Rect rect_roi = get_hbox(rbox, img_h, img_w);

    // 旋转图像
    cv::Mat img_rotate;
    // angle是逆时针从水平转到长边的角度，这里顺时针旋转angle角度，因此长边旋转到水平方向
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, img_rotate, M, cv::Size(img_w, img_h), cv::INTER_LINEAR, 0);


    // Gabor滤波器参数初始化
    int kernel_size;
    double sigma = 1.0, lambd = CV_PI/8, gamma = 0.5, psi = 0;
    // theta 法线方向
    double theta[8];
    int num_theta = 8;
    for (int i=0; i<num_theta; i++)
    {
        theta[i] = (CV_PI/num_theta) * i;
    }

    int num_scale = 8;
    int kernel_size_ls[8] = {3,5,7,9,11,13,15,17};
    // gabor 纹理检测器，可以更多，
    std::vector<cv::Mat> imgs_filtered;
    for (int j=0; j<num_scale; j++)
    {
        kernel_size = kernel_size_ls[j];
        for(int i = 0; i<num_theta; i++)
        {
            cv::Mat kernel1;
            cv::Mat dest;
            kernel1 = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta[i], lambd, gamma, psi, CV_32F);
            filter2D(img_rotate, dest, CV_32F, kernel1);
            imgs_filtered.push_back(dest);

        }
    }

    // 合并为一个多通道图像
    cv::Mat mc_img;
    cv::merge(imgs_filtered, mc_img);

    //
    roi_fe_text = mc_img(rect_roi);

}
void extract_fe_texture(SingleSpectrum &specturm_img, const RBox &rbox_1, const RBox &rbox_2, bool is_fg_bg)
{
    // 提取第一个区域的特征
    cv::Mat img_1 = specturm_img.img_1.clone();
    cv::Mat roi_1_fe_text;
    get_fe_text(img_1, rbox_1, roi_1_fe_text);
    int rows_1, cols_1, chns_1;
    rows_1 = roi_1_fe_text.rows;
    cols_1 = roi_1_fe_text.cols;
    chns_1 = roi_1_fe_text.channels();
    at::Tensor roi_1_tensor = torch::from_blob(
                roi_1_fe_text.data, {rows_1, cols_1, chns_1}, torch::kFloat);




    // 提取第二个区域的特征
    cv::Mat img_2 = specturm_img.img_2.clone();
    cv::Mat roi_2_fe_text;
    get_fe_text(img_2, rbox_2, roi_2_fe_text);
    int rows_2, cols_2, chns_2;
    rows_2 = roi_2_fe_text.rows;
    cols_2 = roi_2_fe_text.cols;
    chns_2 = roi_2_fe_text.channels();
    at::Tensor roi_2_tensor = torch::from_blob(
                roi_2_fe_text.data, {rows_2, cols_2, chns_2}, torch::kFloat);

//    roi_2_tensor_mean = roi_2_tensor_mean - roi_1_tensor_mean;

    at::Tensor roi_1_tensor_mean;
    at::Tensor roi_2_tensor_mean;
    if (is_fg_bg)
    {
        at::Tensor roi_1_tensor_sum = roi_1_tensor.sum(0).sum(0);
        at::Tensor roi_2_tensor_sum = roi_2_tensor.sum(0).sum(0);

        roi_2_tensor_sum = roi_2_tensor_sum - roi_1_tensor_sum;
        roi_1_tensor_mean = roi_1_tensor_sum / (rows_1*cols_1);
        roi_2_tensor_mean = roi_2_tensor_sum / (rows_2*cols_2 - rows_1*cols_1);
    }
    else
    {
        // 每个通道取均值
        roi_1_tensor_mean = roi_1_tensor.mean(0).mean(0);
        roi_2_tensor_mean = roi_2_tensor.mean(0).mean(0);
    }

    specturm_img.roi_fe_1.text = roi_1_tensor_mean;
    specturm_img.roi_fe_2.text = roi_2_tensor_mean;

}

void Probability::extract_fe()
{
    // 每个频谱的图像，分别提取特征
    if (this->img_status.opt)
    {
        ui->text_log->append("提取可见光图像特征...");
        this->extract_fe_SS(this->opt_spectrum);
    }
    if (this->img_status.IR)
    {
        ui->text_log->append("提取红外图像特征...");
        this->extract_fe_SS(this->ir_spectrum);
    }
    if (this->img_status.SAR)
    {
        ui->text_log->append("提取SAR图像特征...");
        this->extract_fe_SS(this->sar_spectrum);
    }
}

// 提取某频谱图像的特征
void Probability::extract_fe_SS(SingleSpectrum &specturm_img)
{

    bool is_fg_bg = this->rois_type == FG_BG;
    // 选择特征
    if (this->fe_status.deep)
    {
        extract_fe_deep(specturm_img, this->rbox_1, this->rbox_2, is_fg_bg);
        ui->text_log->append("   获得深度学习特征！");
    }
    if (this->fe_status.gray)
    {
        extract_fe_gray(specturm_img, this->rbox_1, this->rbox_2, is_fg_bg);
        ui->text_log->append("   获得灰度特征！");
    }
    if (this->fe_status.text)
    {
        extract_fe_texture(specturm_img, this->rbox_1, this->rbox_2, is_fg_bg);
        ui->text_log->append("   获得纹理特征！");
    }
}





// 特征相似度计算和加权
void Probability::on_bu_group_fe_weights(QAbstractButton *button)
{
    QString text = button->text();
    if (text==QString("自定义法")) this->weights_type_fe = CUSTOM;
    else if (text==QString("熵权法")) this->weights_type_fe = ENTROPY;
    else std::cout << "特征加权：没有选择自定义法或熵权法！！！" << std::endl;

    if (this->weights_type_fe == CUSTOM)
    {
        // 切换到自定义法的权值确定界面
        ui->SW_weight_fe->setCurrentWidget(ui->fe_custom);
        // 等待用户输入权值
        ui->le_deep_weight->setText(QString::number(0.0));
        ui->le_gray_weight->setText(QString::number(0.0));
        ui->le_text_weight->setText(QString::number(0.0));
    }
    else if (this->weights_type_fe == ENTROPY)
    {
        ui->SW_weight_fe->setCurrentWidget(ui->fe_entropy);

        // 将熵权法确定的权值显示在屏幕上
        float fe_sim_weights_opt[3] = {0.34179946, 0.34120829, 0.31699225};
        float used_weights[3] = {0.0, 0.0, 0.0};

        if (this->fe_status.deep) used_weights[0] = fe_sim_weights_opt[0];
        if (this->fe_status.gray) used_weights[1] = fe_sim_weights_opt[1];
        if (this->fe_status.text) used_weights[2] = fe_sim_weights_opt[2];

        float sum = used_weights[0] + used_weights[1] + used_weights[2] + 1e-9;
        used_weights[0] /= sum;
        used_weights[1] /= sum;
        used_weights[2] /= sum;

        this->fe_sim_weights.deep = used_weights[0];
        this->fe_sim_weights.gray = used_weights[1];
        this->fe_sim_weights.text = used_weights[2];

        ui->label_deep_weight->setText(QString::number(used_weights[0]));
        ui->label_gray_weight->setText(QString::number(used_weights[1]));
        ui->label_text_weight->setText(QString::number(used_weights[2]));

    }


    if (this->fe_status.deep) {ui->label_deep->show(); ui->le_deep_weight->show();ui->label_deep_2->show(); ui->label_deep_weight->show();}
    else {ui->label_deep->hide(); ui->le_deep_weight->hide();ui->label_deep_2->hide(); ui->label_deep_weight->hide();}

    if (this->fe_status.gray) {ui->label_gray->show(); ui->le_gray_weight->show();ui->label_gray_2->show(); ui->label_gray_weight->show();}
    else {ui->label_gray->hide(); ui->le_gray_weight->hide();ui->label_gray_2->hide(); ui->label_gray_weight->hide();}

    if (this->fe_status.text) {ui->label_text->show(); ui->le_text_weight->show();ui->label_text_2->show(); ui->label_text_weight->show();}
    else {ui->label_text->hide(); ui->le_text_weight->hide();ui->label_text_2->hide(); ui->label_text_weight->hide();}
}

void Probability::reset_weights_fe()
{
    reset_bu_group(this->bu_group_fe_weights, true);

    // 用户输入
    ui->le_deep_weight->setText(QString::number(0.0));
    ui->le_gray_weight->setText(QString::number(0.0));
    ui->le_text_weight->setText(QString::number(0.0));
    // 熵权法
    ui->label_deep_weight->setText(QString::number(0.0));
    ui->label_gray_weight->setText(QString::number(0.0));
    ui->label_text_weight->setText(QString::number(0.0));

    ui->SW_weight_fe->setCurrentWidget(ui->fe_blank);

}

// 频谱相似度计算和加权
void Probability::on_bu_group_sp_weights(QAbstractButton *button)
{
    QString text = button->text();
    if (text==QString("自定义法")) this->weights_type_sp = CUSTOM;
    else if (text==QString("熵权法")) this->weights_type_sp = ENTROPY;
    else std::cout << "频谱加权：没有选择自定义法或熵权法！！！" << std::endl;

    if (this->weights_type_sp == CUSTOM)
    {
        // 切换到自定义法的权值确定界面
        ui->SW_weight_sp->setCurrentWidget(ui->sp_custom);
        // 等待用户输入权值
        ui->le_opt_weight->setText(QString::number(0.0));
        ui->le_IR_weight->setText(QString::number(0.0));
        ui->le_SAR_weight->setText(QString::number(0.0));
    }
    else if (this->weights_type_sp == ENTROPY)
    {
        ui->SW_weight_sp->setCurrentWidget(ui->sp_entropy);

        // 将熵权法确定的权值显示在屏幕上
        float sp_sim_weights[3] = {0.33342042, 0.33320857, 0.33337101};

        float used_weights[3] = {0.0, 0.0, 0.0};
        if (this->img_status.opt) used_weights[0] = sp_sim_weights[0];
        if (this->img_status.IR) used_weights[1] = sp_sim_weights[1];
        if (this->img_status.SAR) used_weights[2] = sp_sim_weights[2];

        float sum = used_weights[0] + used_weights[1] + used_weights[2] + 1e-9;
        used_weights[0] /= sum;
        used_weights[1] /= sum;
        used_weights[2] /= sum;

        this->spectrum_sim_weights.opt = used_weights[0];
        this->spectrum_sim_weights.IR = used_weights[1];
        this->spectrum_sim_weights.SAR = used_weights[2];

        ui->label_opt_weight->setText(QString::number(used_weights[0]));
        ui->label_IR_weight->setText(QString::number(used_weights[1]));
        ui->label_SAR_weight->setText(QString::number(used_weights[2]));
    }

    if (this->img_status.opt) {ui->label_opt->show(); ui->le_opt_weight->show();ui->label_opt_2->show(); ui->label_opt_weight->show();}
    else {ui->label_opt->hide(); ui->le_opt_weight->hide();ui->label_opt_2->hide(); ui->label_opt_weight->hide();}

    if (this->img_status.IR) {ui->label_IR->show(); ui->le_IR_weight->show();ui->label_IR_2->show(); ui->label_IR_weight->show();}
    else {ui->label_IR->hide(); ui->le_IR_weight->hide();ui->label_IR_2->hide(); ui->label_IR_weight->hide();}

    if (this->img_status.SAR) {ui->label_SAR->show(); ui->le_SAR_weight->show();ui->label_SAR_2->show(); ui->label_SAR_weight->show();}
    else {ui->label_SAR->hide(); ui->le_SAR_weight->hide();ui->label_SAR_2->hide(); ui->label_SAR_weight->hide();}

}

void Probability::reset_weights_sp()
{
    reset_bu_group(this->bu_group_sp_weights, true);

    // 用户输入
    ui->le_opt_weight->setText(QString::number(0.0));
    ui->le_IR_weight->setText(QString::number(0.0));
    ui->le_SAR_weight->setText(QString::number(0.0));
    // 熵权法
    ui->label_opt_weight->setText(QString::number(0.0));
    ui->label_IR_weight->setText(QString::number(0.0));
    ui->label_SAR_weight->setText(QString::number(0.0));

    ui->SW_weight_sp->setCurrentWidget(ui->sp_blank);

}

// 特征相似度加权模块初始化，包括特征加权和频谱加权
void Probability::reset_sim_weights()
{

    this->reset_weights_fe();

    this->reset_weights_sp();

    ui->TW_sim_weights->setCurrentWidget(ui->tab_fe);
}

// 巴氏距离
float distance_B(at::Tensor p, at::Tensor q)
{
    // 巴氏系数
    at::Tensor B = torch::sqrt(p*q).sum();
    // 巴氏距离
    float a = torch::log(B).item().toFloat();
    return a;
}
float cal_similarity_deep(const at::Tensor &roi_fe_1, const at::Tensor &roi_fe_2)
{
    float sim = 0.0;

    // 采用欧氏距离
    float diff = (roi_fe_1 - roi_fe_2).norm(2).item().toFloat();
//    std::cout << "diff 1:" << diff << std::endl;
    // 再除以向量的长度
    diff /= roi_fe_1.sizes()[0];
//    std::cout << "diff 2:" << diff << std::endl;
    // 欧氏距离的范围为[0~无穷)，使用exp(-x)，映射到0~1之间，即为特征相似度

    // 让深度学习特征的欧氏距离分布中心0.002，映射到特征相似度0.5
    sim = exp(346.575*(-diff));

    return sim;

}

float cal_similarity_gray(const at::Tensor &roi_fe_1, const at::Tensor &roi_fe_2)
{
    // 灰度直方图，使用巴氏系数作为特征相似度

    return 0.0;

}

float cal_similarity_text(const at::Tensor &roi_fe_1, const at::Tensor &roi_fe_2)
{
    float sim = 0.0;
//    float diff = torch::abs(roi_fe_1 - roi_fe_2).sum().item().toFloat();

//    float norm_1 = torch::abs(roi_fe_1).sum().item().toFloat();
//    float norm_2 = torch::abs(roi_fe_2).sum().item().toFloat();


//    sim = diff / (norm_1 + norm_2 - diff);
////    std::cout << " text diff sum:" << diff.sum().item().toFloat() << " text diff mean:" << diff.mean().item().toFloat() << std::endl;
//    sim = 1 - sim;

    // 采用欧氏距离
    float diff = (roi_fe_1 - roi_fe_2).norm(2).item().toFloat();
//    std::cout << "diff 1:" << diff << std::endl;
    // 再除以向量的长度
    diff /= roi_fe_1.sizes()[0];
//    std::cout << "diff 2:" << diff << std::endl;
    // 欧氏距离的范围为[0~无穷)，使用exp(-x)，映射到0~1之间，即为特征相似度
    sim = exp(-diff);


    return sim;

}

void Probability::cal_similarity()
{

    // 读取特征相似度加权权值
    if (this->weights_type_fe==CUSTOM)
    {
        this->fe_sim_weights.deep = ui->le_deep_weight->text().toFloat();
        this->fe_sim_weights.gray = ui->le_gray_weight->text().toFloat();
        this->fe_sim_weights.text = ui->le_text_weight->text().toFloat();
    }
    else if (this->weights_type_fe==ENTROPY)
    {

        this->fe_sim_weights.deep = ui->label_deep_weight->text().toFloat();
        this->fe_sim_weights.gray = ui->label_gray_weight->text().toFloat();
        this->fe_sim_weights.text = ui->label_text_weight->text().toFloat();

    }

    // 判断权值之和是否为1
    float eps = 1e-8;
    float sum = this->fe_sim_weights.deep + this->fe_sim_weights.gray + this->fe_sim_weights.text;
    if (fabs(sum-1.0) > eps)
    {
        ui->text_log->append(QString("特征权值之和不为1，请重新配置权值..."));
    }

    // 计算每个频谱的图像的特征相似度
    if (this->img_status.opt)
    {
        ui->text_log->append("提取可见光图像相似度...");
        this->cal_similarity_SS(this->opt_spectrum);
    }
    if (this->img_status.IR)
    {
        ui->text_log->append("提取红外图像相似度...");
        this->cal_similarity_SS(this->ir_spectrum);
    }
    if (this->img_status.SAR)
    {
        ui->text_log->append("提取SAR图像相似度...");
        this->cal_similarity_SS(this->sar_spectrum);
    }



    // 读取特征相似度加权权值
    if (this->weights_type_sp==CUSTOM)
    {
        this->spectrum_sim_weights.opt = ui->le_opt_weight->text().toFloat();
        this->spectrum_sim_weights.IR = ui->le_IR_weight->text().toFloat();
        this->spectrum_sim_weights.SAR = ui->le_SAR_weight->text().toFloat();
    }
    else if (this->weights_type_sp==ENTROPY)
    {
        this->spectrum_sim_weights.opt = ui->label_opt_weight->text().toFloat();
        this->spectrum_sim_weights.IR = ui->label_IR_weight->text().toFloat();
        this->spectrum_sim_weights.SAR = ui->label_SAR_weight->text().toFloat();
    }


    sum = this->spectrum_sim_weights.opt + this->spectrum_sim_weights.IR + this->spectrum_sim_weights.SAR;
    if (fabs(sum-1.0) > eps)
    {
        ui->text_log->append(QString("频谱权值之和不为1，请重新配置权值..."));
    }


    float si = 0;
    if (this->img_status.opt) si = si + this->opt_spectrum.sim * this->spectrum_sim_weights.opt;
    if (this->img_status.IR) si = si + this->ir_spectrum.sim * this->spectrum_sim_weights.IR;
    if (this->img_status.SAR) si = si + this->sar_spectrum.sim * this->spectrum_sim_weights.SAR;

    this->similarity = si;

    QString log = QString("\n综合特征相似度为:") + QString::number(this->similarity, 'f', 3);
    ui->text_log->append(log);

}

// 提取某频谱图像的特征相似度
void Probability::cal_similarity_SS(SingleSpectrum &specturm_img)
{
    if (this->fe_status.deep)
    {
        specturm_img.fe_similarity.deep = cal_similarity_deep(specturm_img.roi_fe_1.deep, specturm_img.roi_fe_2.deep);
        QString log = QString("深度学习特征相似度为:") + QString::number(specturm_img.fe_similarity.deep, 'f', 3);
        ui->text_log->append(log);

    }
    if (this->fe_status.gray)
    {
        float a = cal_similarity_gray(specturm_img.roi_fe_1.gray, specturm_img.roi_fe_2.gray);
        QString log = QString("灰度特征相似度为:") + QString::number(specturm_img.fe_similarity.gray, 'f', 3);
        ui->text_log->append(log);
    }
    if (this->fe_status.text)
    {
        specturm_img.fe_similarity.text = cal_similarity_text(specturm_img.roi_fe_1.text, specturm_img.roi_fe_2.text);
        QString log = QString("纹理特征相似度为:") + QString::number(specturm_img.fe_similarity.text, 'f', 3);
        ui->text_log->append(log);
    }


    float si = 0;
    if (this->fe_status.deep) si = si + specturm_img.fe_similarity.deep * this->fe_sim_weights.deep;
    if (this->fe_status.gray) si = si + specturm_img.fe_similarity.gray * this->fe_sim_weights.gray;
    if (this->fe_status.text) si = si + specturm_img.fe_similarity.text * this->fe_sim_weights.text;

    specturm_img.sim = si;

}





// 特征相似度到识别概率的映射

// 心理学决策原理法
float sim_map_pro_psychology_fg_bg(float similarity)
{

    float k = -4.290018580259924;
    float p = 0.7416156711588913;

    float probability;
    float temp = -k * (similarity-p);
    probability= 1 - exp(temp);

    if (probability >1.0 || probability <0.0) probability = 1-similarity;

    return probability;
}
float sim_map_pro_psychology_fg_fg(float similarity)
{

    float k = -0.9135134583025529;
    float p = 0.9450706058862101;

    float probability;
    float temp = -k * (similarity-p);
    probability= 1 - exp(temp);

    if (probability >1.0 || probability <0.0) probability = 1-similarity;

    return probability;
}
void Probability::cal_probability()
{

    // 由综合特征相似度计算识别概率，有函数映射法和心理学决策原理法，共2种方法
    if (this->rois_type==0) this->probability = sim_map_pro_psychology_fg_bg(this->similarity);
    else this->probability = sim_map_pro_psychology_fg_fg(this->similarity);
    QString log = QString("\n计算得到的识别概率为:") + QString::number(this->probability, 'f', 3);
    ui->text_log->append(log);
}




void Probability::browse_save()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    // qDebug() << path;
    ui->le_savepath->setText(path);
}

void Probability::save_results()
{
    QString path = ui->le_savepath->text();
    QString txt_name = ui->le_savename->text();

    // 从右向左查找，如果没找到，说明没有后缀
    if(txt_name.lastIndexOf(".txt") == -1) txt_name += ".txt";

    QString txt_save_path = path +"/"+ txt_name;

    std::ofstream fout;
    fout.open(txt_save_path.toStdString());


    // 把频谱图像写入
    fout << "频谱图像：\n";
    if (this->img_status.opt) fout << "  可见光：img_1 " << this->opt_spectrum.img_1_path << "，img_2 " << this->opt_spectrum.img_2_path << "\n";
    if (this->img_status.IR) fout << "  红外：img_1 " << this->ir_spectrum.img_1_path << "，img_2 " << this->ir_spectrum.img_2_path << "\n";
    if (this->img_status.SAR) fout << "  SAR：img_1 " << this->sar_spectrum.img_1_path << "，img_2 " << this->sar_spectrum.img_2_path << "\n";



    if (this->rois_type==FG_BG) fout << "目标-背景\n";
    if (this->rois_type==FG_FG) fout << "目标-目标（单张图像）\n";
    if (this->rois_type==FG_FG2) fout << "目标-目标2（两张图像）\n";
    // 保存两个提取区域的目标框
    fout << "  box_1:";
    std::string point, line;
    point.clear();
    line.clear();
    for (int j=0; j<4; j++)
    {
        int x = this->contour_1[j].x;
        int y = this->contour_1[j].y;
        if (j<3) point = std::to_string(x) + ',' + std::to_string(y) + ',';
        else point = std::to_string(x) + ',' + std::to_string(y) + '\n';

        line = line + point;
    }
    fout << line;
    fout << "  box_2:";
    point.clear();
    line.clear();
    for (int j=0; j<4; j++)
    {
        int x = this->contour_2[j].x;
        int y = this->contour_2[j].y;
        if (j<3) point = std::to_string(x) + ',' + std::to_string(y) + ',';
        else point = std::to_string(x) + ',' + std::to_string(y) + '\n';

        line = line + point;
    }
    fout << line;




    // 写入选择的图像特征，和特征加权权值
    fout << "图像特征加权: \n";
    if (this->fe_status.deep) fout << "  深度学习特征权值：" << this->fe_sim_weights.deep << "\n";
    if (this->fe_status.gray) fout << "  灰度特征权值：" << this->fe_sim_weights.gray << "\n";
    if (this->fe_status.text) fout << "  纹理特征权值：" << this->fe_sim_weights.text << "\n";
    fout << "频谱加权: \n";
    if (this->img_status.opt) fout << "  可见光权值：" << this->spectrum_sim_weights.opt << "\n";
    if (this->img_status.IR) fout << "  红外权值：" << this->spectrum_sim_weights.IR << "\n";
    if (this->img_status.SAR) fout << "  SAR权值：" << this->spectrum_sim_weights.SAR << "\n";


    std::string s = "特征相似度:" + std::to_string(this->similarity) + "\n";
    fout << s;
    s = "目标识别概率:" + std::to_string(this->probability) + "\n";
    fout << s;

    fout.close();

    ui->text_log->append("已完成保存！！！");

}



void Probability::init_ui()
{

    // 频谱图像配置界面初始化
    this->reset_MS();

    // rois区域初始化
    this->reset_rois();

    // 特征选择和提取界面初始化
    this->reset_fe();

    // 相似度加权计算
    this->reset_sim_weights();


    // 软件日志初始化
    ui->text_log->clear();
    ui->text_log->setText("请选择频谱图像...");
    // 保存路径
    ui->le_savepath->clear();
    ui->le_savename->clear();

}
