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

void draw_rboxes(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours,
                 int contoursIds=-1, cv::Scalar color = cv::Scalar(0,0,255), int thickness=3)
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
void draw_rboxes_ids(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours,
                     cv::Scalar color = cv::Scalar(0,0,255), int thickness=3)
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
        cv::putText(img_result, box_id, point, fontFace, fontScale, color, thickness=thickness);
    }

}


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
    connect(ui->CB_roi_choose_1, &QComboBox::currentTextChanged, this, &Probability::show_all_objs_img_1);
    connect(ui->CB_roi_choose_2, &QComboBox::currentTextChanged, this, &Probability::show_all_objs_img_1);
    connect(ui->CB_roi_choose_3, &QComboBox::currentTextChanged, this, &Probability::show_all_objs_img_1);
    connect(ui->CB_roi_choose_4, &QComboBox::currentTextChanged, this, &Probability::show_all_objs_img_1);
    connect(ui->CB_roi_choose_5, &QComboBox::currentTextChanged, this, &Probability::show_all_objs_img_2);


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
    bu_group_weights.setParent(this);
    bu_group_weights.setExclusive(true);
    bu_group_weights.addButton(ui->RB_weights_custom, 0);
    bu_group_weights.addButton(ui->RB_weights_entropy, 1);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_weights, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_group_weights);

    // 映射方式选择按钮组
    bu_group_map.setParent(this);
    bu_group_map.setExclusive(true);
    bu_group_map.addButton(ui->RB_map_func, 0);
    bu_group_map.addButton(ui->RB_map_psychology, 1);
    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_group_map, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_group_map);


    this->init_ui();

}

Probability::~Probability() {delete ui;}

void Probability::reset_show()
{
    this->reset_imgpath_show();
    this->reset_img_show();
}

void Probability::reset_imgpath_show()
{

    ui->le_imgpath_opt->clear();
    ui->le_imgpath_IR->clear();
    ui->le_imgpath_SAR->clear();


    ui->widget_opt->hide();
    ui->widget_IR->hide();
    ui->widget_SAR->hide();
}
void Probability::reset_img_show()
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
void Probability::on_bu_group_MS(QAbstractButton *button)
{

    this->reset_show();

    // 状态归零
    this->img_status = {false, false, false};

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

    this->reset_imgpath_show();

    if (this->img_status.opt) ui->widget_opt->show();
    if (this->img_status.IR)  ui->widget_IR->show();
    if (this->img_status.SAR) ui->widget_SAR->show();


    // 区域选择也reset
    this->reset_rois();

}

void Probability::reset_bu_group_MS()
{

    // 遍历按钮，全部设置为未选中状态
    QList<QAbstractButton*> list;

    // 按钮组不互斥
    this->bu_group_MS.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    list = this->bu_group_MS.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);

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
    this->rois_type = button->text();
    this->reset_rois_show();

    if (this->rois_type == QString("目标-背景")) ui->SW_choose_rois->setCurrentWidget(ui->page_fg_bg);
    if (this->rois_type == QString("目标-目标")) ui->SW_choose_rois->setCurrentWidget(ui->page_fg_fg);

    if (this->rois_type == QString("目标-背景") || this->rois_type == QString("目标-目标"))
    {
        ui->SW_choose_rois->setCurrentWidget(ui->page_fg_bg);

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


    if (this->rois_type == QString("目标-目标2"))
    {
        ui->SW_choose_rois->setCurrentWidget(ui->page_fg_fg2);

        if (this->img_status.opt) ui->widget_opt_2->show();
        if (this->img_status.IR) ui->widget_IR_2->show();
        if (this->img_status.SAR) ui->widget_SAR_2->show();
    };


}
void Probability::reset_bu_group_rois()
{
    // 按钮组中不互斥
    this->bu_group_rois.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    QList<QAbstractButton*> list;
    list = this->bu_group_rois.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);
    // 按钮组互斥
    this->bu_group_rois.setExclusive(true);
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

    ui->le_bg_ratio->setText(QString::number(1.0));

    // 如果QComboBox只有一个-1，此时clear会导致text发生变化，从而导致信号触发，因此在clear前先断开信号连接
    reset_QCB(ui->CB_roi_choose_1);
    reset_QCB(ui->CB_roi_choose_2);
    reset_QCB(ui->CB_roi_choose_3);
    reset_QCB(ui->CB_roi_choose_4);
    reset_QCB(ui->CB_roi_choose_5);

    // 设置label的大小
    int h=400,w=400;
    ui->labelImage_opt_2->resize(w,h);
    ui->labelImage_IR_2->resize(w,h);
    ui->labelImage_SAR_2->resize(w,h);

    ui->labelImage_opt_2->clear();
    ui->labelImage_IR_2->clear();
    ui->labelImage_SAR_2->clear();
    ui->widget_opt_2->hide();
    ui->widget_IR_2->hide();
    ui->widget_SAR_2->hide();

}

void Probability::reset_rois()
{
    this->reset_bu_group_rois();
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

    this->show_all_objs_img_1(QString(" "));

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

    this->show_all_objs_img_1(QString(" "));


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

    this->show_all_objs_img_1(QString(" "));

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
    this->show_all_objs_img_2(QString(" "));

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

void Probability::show_all_objs_img_1(const QString &text)
{
    if (text == QString(" "))
    {

        if (this->img_status.opt)
        {
            cv::Mat img_result;
            // 画出所有框
            draw_rboxes_ids(this->opt_spectrum.img_1, img_result, this->contours_1);
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
            //显示图像
            QImage srcimg = MatToImage(img_result);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg.scaled(ui->labelImage_SAR->size(),Qt::KeepAspectRatio);
            ui->labelImage_SAR->setPixmap(QPixmap::fromImage(dest));
        }

    }
}

void Probability::show_all_objs_img_2(const QString &text)
{
    if (text == QString(" "))
    {
        if (this->img_status.opt)
        {
            cv::Mat img_result;
            // 画出所有框
            draw_rboxes_ids(this->opt_spectrum.img_2, img_result, this->contours_2);
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
            //显示图像
            QImage srcimg = MatToImage(img_result);
            // 图像缩放到label的大小，并保持长宽比
            QImage dest = srcimg.scaled(ui->labelImage_SAR_2->size(),Qt::KeepAspectRatio);
            ui->labelImage_SAR_2->setPixmap(QPixmap::fromImage(dest));
        }
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
    std::vector<std::vector<cv::Point>> contour_1;
    std::vector<std::vector<cv::Point>> contour_2;
    if (this->rois_type == QString("目标-背景"))
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

        cv::Point2f center = fg_rbox.center;
        // 单位是角度
        float angle = fg_rbox.angle;
        float w = fg_rbox.size.width;
        float h = fg_rbox.size.height;

        // 背景区域
        float padding = bg_ratio * h / 2;
        float bg_w = w+2*padding;
        float bg_h = h+2*padding;
//        cv::RotatedRect bg_rbox = cv::RotatedRect(center, cv::Size2f(bg_w,bg_h), angle);

        RBox bg_rbox = {center, cv::Size2f(bg_w,bg_h), angle};

        this->contour_2.clear();
        rbox2points(this->contour_2, bg_rbox);
        contour_2.clear();
        contour_2.push_back(this->contour_2);


        // 画出选中的roi
        this->show_rois_img_1(contour_1, contour_2);


    }

    else if (this->rois_type == QString("目标-目标"))
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

        // 画出选中的roi
        this->show_rois_img_1(contour_1, contour_2);

    }

    else if (this->rois_type == QString("目标-目标2"))
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
}

void Probability::reset_bu_group_fe()
{
    // 按钮组中不互斥
    this->bu_group_fe.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    QList<QAbstractButton*> list;
    list = this->bu_group_fe.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);
}

int map_roi_levels(float roi_scale, int num_levels, int finest_scale=56)
{

    int level_id = floor(log2(roi_scale / finest_scale + 1e-6));
    level_id = (level_id<0) ? 0 : level_id;
    level_id = (level_id>(num_levels-1)) ? (num_levels-1) : level_id;

    return level_id;
}

void get_fe_deep(cv::Mat &img, const at::Tensor &rbox, at::Tensor &roi_fe_deep)
{
    std::string model_path = "./model/extract_fe/exp361_no_align_extract_fe_script.pt";

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

void extract_fe_deep(SingleSpectrum &specturm_img, const std::vector<cv::Point> &contour_1, const std::vector<cv::Point> &contour_2)
{

    // 提取第一个区域的特征
    cv::Mat img_1 = specturm_img.img_1.clone();
    at::Tensor rois_1;
    points2xywhtheta(contour_1, rois_1);
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
    get_fe_deep(img_1, rois_1, roi_1_fe);


    // 提取第二个区域的特征
    cv::Mat img_2 = specturm_img.img_2.clone();
    at::Tensor rois_2;
    points2xywhtheta(contour_2, rois_2);
//    rois_2[4] = -0.3122;
    // 只有一张图像
    rois_2 = torch::cat({at::tensor({batch_id}), rois_2}, 0);
    // shape [N, 6(batch_id, x(像素单位), y, w, h, theta(弧度))]
    rois_2 = rois_2.unsqueeze(0);

    at::Tensor roi_2_fe;
    get_fe_deep(img_2, rois_2, roi_2_fe);



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
void extract_fe_gray(SingleSpectrum &specturm_img, const std::vector<cv::Point> &contour_1, const std::vector<cv::Point> &contour_2)
{

//    std::cout << "box1:" << this->contour_1 << std::endl;
//    std::cout << "box2:" << this->contour_2 << std::endl;

    // 提取第一个区域的特征
    cv::Mat img_1 = specturm_img.img_1.clone();
    RBox rbox_1;
    points2rbox(contour_1, rbox_1);
    cv::Mat roi_hist_1;
    get_fe_gray(img_1, rbox_1, roi_hist_1);


    // 提取第二个区域的特征
    cv::Mat img_2 = specturm_img.img_2.clone();
    RBox rbox_2;
    points2rbox(contour_2, rbox_2);
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
    int kernel_size = 3;
    double sigma = 1.0, lambd = CV_PI/8, gamma = 0.5, psi = 0;
    // theta 法线方向
    double theta[8];
    int num_theta = 8;
    for (int i=0; i<num_theta; i++)
    {
        theta[i] = (CV_PI/num_theta) * i;
    }
    // gabor 纹理检测器，可以更多，
    std::vector<cv::Mat> imgs_filtered;
    for(int i = 0; i<num_theta; i++)
    {
        cv::Mat kernel1;
        cv::Mat dest;
        kernel1 = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta[i], lambd, gamma, psi, CV_32F);
        filter2D(img_rotate, dest, CV_32F, kernel1);
        imgs_filtered.push_back(dest);

    }
    // 合并为一个多通道图像
    cv::Mat mc_img;
    cv::merge(imgs_filtered, mc_img);

    //
    roi_fe_text = mc_img(rect_roi);

}
void extract_fe_texture(SingleSpectrum &specturm_img, const std::vector<cv::Point> &contour_1, const std::vector<cv::Point> &contour_2)
{
    // 提取第一个区域的特征
    cv::Mat img_1 = specturm_img.img_1.clone();
    RBox rbox_1;
    points2rbox(contour_1, rbox_1);
    cv::Mat roi_1_fe_text;
    get_fe_text(img_1, rbox_1, roi_1_fe_text);
    at::Tensor roi_1_tensor = torch::from_blob(
                roi_1_fe_text.data, {roi_1_fe_text.rows, roi_1_fe_text.cols, roi_1_fe_text.channels()}, torch::kFloat);
    // 每个通道取均值
    at::Tensor roi_1_tensor_mean = roi_1_tensor.mean(0).mean(0);



    // 提取第二个区域的特征
    cv::Mat img_2 = specturm_img.img_2.clone();
    RBox rbox_2;
    points2rbox(contour_2, rbox_2);
    cv::Mat roi_2_fe_text;
    get_fe_text(img_2, rbox_2, roi_2_fe_text);
    at::Tensor roi_2_tensor = torch::from_blob(
                roi_2_fe_text.data, {roi_2_fe_text.rows, roi_2_fe_text.cols, roi_2_fe_text.channels()}, torch::kFloat);
    at::Tensor roi_2_tensor_mean = roi_2_tensor.mean(0).mean(0);
//    roi_2_tensor_mean = roi_2_tensor_mean - roi_1_tensor_mean;


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

    // 选择特征
    if (this->fe_status.deep)
    {
        extract_fe_deep(specturm_img, this->contour_1, this->contour_2);
        ui->text_log->append("   获得深度学习特征！");
    }
    if (this->fe_status.gray)
    {
        extract_fe_gray(specturm_img, this->contour_1, this->contour_2);
        ui->text_log->append("   获得灰度特征！");
    }
    if (this->fe_status.text)
    {
        extract_fe_texture(specturm_img, this->contour_1, this->contour_2);
        ui->text_log->append("   获得纹理特征！");
    }
}







// 特征相似度计算和加权
void Probability::on_bu_group_weights(QAbstractButton *button)
{
    this->weights_type = button->text();
    if (this->weights_type == QString("自定义法"))
    {
        // 等待用户输入权值
        // 将熵权法确定的权值显示在屏幕上
        ui->le_deep_weight->setText(QString::number(0.0));
        ui->le_gray_weight->setText(QString::number(0.0));
        ui->le_text_weight->setText(QString::number(0.0));
    }
    else if (this->weights_type == QString("熵权法"))
    {
        // 将熵权法确定的权值显示在屏幕上
        ui->le_deep_weight->setText(QString::number(0.5));
        ui->le_gray_weight->setText(QString::number(0.3));
        ui->le_text_weight->setText(QString::number(0.2));
    }
}

void Probability::reset_weights_show()
{
    ui->le_deep_weight->setText(QString::number(0.0));
    ui->le_gray_weight->setText(QString::number(0.0));
    ui->le_text_weight->setText(QString::number(0.0));
}

void Probability::reset_bu_group_weights()
{
    // 按钮组中不互斥
    this->bu_group_weights.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    QList<QAbstractButton*> list;
    list = this->bu_group_weights.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);
    // 按钮组互斥
    this->bu_group_weights.setExclusive(true);
}

void Probability::reset_weights()
{
    this->reset_bu_group_weights();
    this->reset_weights_show();
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
    sim = exp(-diff);

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
    float diff = torch::abs(roi_fe_1 - roi_fe_2).sum().item().toFloat();

    float norm_1 = torch::abs(roi_fe_1).sum().item().toFloat();
    float norm_2 = torch::abs(roi_fe_2).sum().item().toFloat();


    sim = diff / (norm_1 + norm_2 - diff);
//    std::cout << " text diff sum:" << diff.sum().item().toFloat() << " text diff mean:" << diff.mean().item().toFloat() << std::endl;
    sim = 1 - sim;

//    // 灰度直方图，使用巴氏系数作为特征相似度
//    float similarity = torch::sqrt(this->roi_fe_1.text*this->roi_fe_2.text).sum().item().toFloat();

    return sim;

}

void Probability::cal_similarity()
{

    // 读取特征相似度加权权值
    this->fe_similarity_weights.deep = ui->le_deep_weight->text().toFloat();
    this->fe_similarity_weights.gray = ui->le_gray_weight->text().toFloat();
    this->fe_similarity_weights.text = ui->le_text_weight->text().toFloat();


    // 计算每个频谱的图像的特征相似度

    // 每个频谱的图像，分别提取特征
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



    this->img_sim_weights = {0.5,0.3,0.2};
    float si = 0;
    if (this->img_status.opt) si = si + this->opt_spectrum.sim * this->img_sim_weights.opt;
    if (this->img_status.IR) si = si + this->ir_spectrum.sim * this->img_sim_weights.IR;
    if (this->img_status.SAR) si = si + this->sar_spectrum.sim * this->img_sim_weights.SAR;

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
    if (this->fe_status.deep) si = si + specturm_img.fe_similarity.deep * this->fe_similarity_weights.deep;
    if (this->fe_status.gray) si = si + specturm_img.fe_similarity.gray * this->fe_similarity_weights.gray;
    if (this->fe_status.text) si = si + specturm_img.fe_similarity.text * this->fe_similarity_weights.text;

    specturm_img.sim = si;

}





// 特征相似度到识别概率的映射
// 识别概率映射
void Probability::on_bu_group_map(QAbstractButton *button) {this->map_type = button->text();}

void Probability::reset_bu_group_map()
{
    // 按钮组中不互斥
    this->bu_group_map.setExclusive(false);
    // 遍历按钮，全部设置为未选中状态
    QList<QAbstractButton*> list;
    list = this->bu_group_map.buttons();
    foreach (QAbstractButton *button, list) button->setChecked(false);
    // 按钮组互斥
    this->bu_group_map.setExclusive(true);if(this->img_status.opt) this->opt_spectrum.img_2 = this->opt_spectrum.img_1.clone();
    if(this->img_status.IR) this->ir_spectrum.img_2 = this->ir_spectrum.img_1.clone();
    if(this->img_status.SAR) this->sar_spectrum.img_2 = this->sar_spectrum.img_1.clone();

}
// 函数映射法
float sim_map_pro_func_map(float similarity)
{
    float probability;
    probability= 1 - similarity;

    return probability;
}

// 心理学决策原理法
float sim_map_pro_psychology(float similarity)
{
    // 以下几组参数，来自论文《基于图像特征与心理感知量的伪装效果评价方法》
    // 灰度特征
    float k = 3.965;
    float p = -0.024;
    float q = 1.333;

//    // 色度特征
//    float k = 1.731;
//    float p = -0.030;
//    float q = 1.048;

//    // 纹理特征
//    float k = 0.464;
//    float p = -0.002;
//    float q = 0.667;

    float probability;
    float temp = -k * pow((similarity-p), q);
    probability= 1 - exp(temp);

    return probability;
}

void Probability::cal_probability()
{

    // 由综合特征相似度计算识别概率，有函数映射法和心理学决策原理法，共2种方法
    QString map_type = this->map_type;

    if (map_type == QString("函数映射法"))
    {
        this->probability = sim_map_pro_func_map(this->similarity);
    }
    else if (map_type == QString("心理学决策原理法"))
    {
        this->probability = sim_map_pro_psychology(this->similarity);
    }
    else
    {
        std::cout << "没有选中相似度-识别概率映射方法" << std::endl;
        this->probability = -1.0;
    }

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


    fout << this->rois_type.toStdString() + "\n";

    std::string s = "特征相似度:" + std::to_string(this->similarity) + "\n";
    fout << s;
    s = "目标识别概率:" + std::to_string(this->probability) + "\n";
    fout << s;

    fout.close();

    ui->text_log->append("已完成保存！！！");

}



void Probability::init_ui()
{
    // 初始化图像路径界面和图像显示界面
    this->reset_show();
    this->reset_bu_group_MS();


    this->reset_rois();
    this->reset_bu_group_fe();
    this->reset_weights();
    this->reset_bu_group_map();

    // 软件日志初始化
    ui->text_log->clear();
    ui->text_log->setText("请选择频谱图像...");
    // 保存路径
    ui->le_savepath->clear();

}
