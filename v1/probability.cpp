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

void draw_rboxes_ids(const cv::Mat &img, cv::Mat &img_result, const std::vector<std::vector<cv::Point>> &contours,
                     cv::Scalar color = cv::Scalar(0,0,255), int thickness=3)
{

    img_result = img.clone();

    int num_boxes = contours.size();
    for (int id=0; id < num_boxes; id++)
    {
        // 画一个框
        cv::drawContours(img_result, contours, id, color, thickness);


//        putText( InputOutputArray img, const String& text, Point org,
//                                 int fontFace, double fontScale, Scalar color,
//                                 int thickness = 1, int lineType = LINE_8,
//                                 bool bottomLeftOrigin = false )
        // 画出boxes的编号
        std::string box_id = std::to_string(id);

        cv::Point point = contours[id][0];
        int fontFace = cv::FONT_HERSHEY_COMPLEX;        // 字体类型
        double fontScale=2.0;    // 字体大小
        cv::putText(img_result, box_id, point, fontFace, fontScale, color, thickness=thickness);
    }

}


Probability::Probability(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Probability)
{
    ui->setupUi(this);
    connect(ui->bu_exit, &QPushButton::clicked, this, &QMainWindow::close);
    // 读取图像文件路径
    connect(ui->bu_browse_1, &QPushButton::clicked, this, &Probability::browse_img_1);
    // 读取模型文件路径
    connect(ui->bu_browse_gt_1, &QPushButton::clicked, this, &Probability::browse_gt_1);

    // 读取图像文件路径
    connect(ui->bu_browse_2, &QPushButton::clicked, this, &Probability::browse_img_2);
    // 读取模型文件路径
    connect(ui->bu_browse_gt_2, &QPushButton::clicked, this, &Probability::browse_gt_2);

    // 读取图像文件路径
    connect(ui->bu_browse_3, &QPushButton::clicked, this, &Probability::browse_img_3);
    // 读取模型文件路径
    connect(ui->bu_browse_gt_3, &QPushButton::clicked, this, &Probability::browse_gt_3);

    // 读取图像文件路径
    connect(ui->bu_browse_4, &QPushButton::clicked, this, &Probability::browse_img_4);
    // 读取模型文件路径
    connect(ui->bu_browse_gt_4, &QPushButton::clicked, this, &Probability::browse_gt_4);


    // 目标选择
    connect(ui->CB_roi_choose_1, &QComboBox::currentTextChanged, this, &Probability::choose_roi_1);
    connect(ui->CB_roi_choose_2, &QComboBox::currentTextChanged, this, &Probability::choose_roi_2);
    connect(ui->CB_roi_choose_3, &QComboBox::currentTextChanged, this, &Probability::choose_roi_3);
    connect(ui->CB_roi_choose_4, &QComboBox::currentTextChanged, this, &Probability::choose_roi_4);
    connect(ui->CB_roi_choose_5, &QComboBox::currentTextChanged, this, &Probability::choose_roi_5);


    // 提取特征
    connect(ui->bu_extract, &QPushButton::clicked, this, &Probability::extract_fe);


    connect(ui->bu_similarity, &QPushButton::clicked, this, &Probability::cal_similarity);

    connect(ui->bu_probability, &QPushButton::clicked, this, &Probability::cal_probability);

    connect(ui->bu_save, &QPushButton::clicked, this, &Probability::save_results);
    // 显示软件日志
    ui->text_log->setText("请输入图像路径和标注文件路径...");

    // 保存路径
    connect(ui->bu_browse_save, &QPushButton::clicked, this, &Probability::browse_save);

    // 背景区域
    connect(ui->le_bg_ratio, &QLineEdit::textChanged, this, &Probability::change_bg_ratio);


    // 特征相似度加权权值的手动输入
    connect(ui->CB_weights, &QComboBox::currentTextChanged, this, &Probability::show_CB_weights);

    // 设置label的大小
    int h=512,w=512;
    ui->labelImage_1->resize(w,h);
    ui->labelImage_2->resize(w,h);





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


    // 特征提取区域选择按钮组
    bu_rois_group.setParent(this);
    // 按钮组互斥
    bu_rois_group.setExclusive(true);
    bu_rois_group.addButton(ui->RB_fg_bg, 0);
    bu_rois_group.addButton(ui->RB_fg_fg, 1);
    bu_rois_group.addButton(ui->RB_fg_fg2, 2);

    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_rois_group, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_rois_group);


}

Probability::~Probability()
{
    delete ui;
}


void Probability::browse_img(QString type)
{
    QString img_path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Images(*.png *.jpg)"
                );

    if (type == QString("目标-背景")) ui->le_imgpath_1->setText(img_path);
    if (type == QString("目标-目标")) ui->le_imgpath_2->setText(img_path);
    if (type == QString("目标-目标2")) ui->le_imgpath_3->setText(img_path);

    // 加载图像
    LoadImage(img_path.toStdString(), this->img_1); // CV_8UC3

    //显示图像
    QImage* srcimg = new QImage;
    srcimg->load(img_path);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg->scaled(ui->labelImage_1->size(),Qt::KeepAspectRatio);
    ui->labelImage_1->setPixmap(QPixmap::fromImage(dest));

}
void Probability::browse_img_1()
{
    browse_img(QString("目标-背景"));
}
void Probability::browse_img_2()
{
    browse_img(QString("目标-目标"));
}
void Probability::browse_img_3()
{
    browse_img(QString("目标-目标2"));
}
void Probability::browse_img_4()
{
    QString img_path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Images(*.png *.jpg)"
                );

    ui->le_imgpath_4->setText(img_path);

    // 加载图像
    LoadImage(img_path.toStdString(), this->img_2); // CV_8UC3

    //显示图像
    QImage* srcimg = new QImage;
    srcimg->load(img_path);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg->scaled(ui->labelImage_2->size(),Qt::KeepAspectRatio);
    ui->labelImage_2->setPixmap(QPixmap::fromImage(dest));
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

    cv::Mat img_result;
    // 画出所有框
    draw_rboxes_ids(this->img_1, img_result, this->contours_1);
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_1->size(),Qt::KeepAspectRatio);
    ui->labelImage_1->setPixmap(QPixmap::fromImage(dest));

    // 可选的框展示给用户
    ui->CB_roi_choose_1->clear();
    int num_boxes = this->contours_1.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_1->addItem(QString::number(i));
    }
    ui->CB_roi_choose_1->addItem(QString::number(-1));
    ui->CB_roi_choose_1->setCurrentIndex(num_boxes); //显示-1

}

void Probability::change_bg_ratio(const QString text)
{
    QString text_roi_id = ui->CB_roi_choose_1->currentText();
    this->choose_roi_1(text_roi_id);
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

    cv::Mat img_result;
    // 画出所有框
    draw_rboxes_ids(this->img_1, img_result, this->contours_1);
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_1->size(),Qt::KeepAspectRatio);
    ui->labelImage_1->setPixmap(QPixmap::fromImage(dest));

    // 可选的框展示给用户
    ui->CB_roi_choose_2->clear();
    ui->CB_roi_choose_3->clear();
    int num_boxes = this->contours_1.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_2->addItem(QString::number(i));
        ui->CB_roi_choose_3->addItem(QString::number(i));
    }
    ui->CB_roi_choose_2->addItem(QString::number(-1));
    ui->CB_roi_choose_2->setCurrentIndex(num_boxes); //显示-1
    ui->CB_roi_choose_3->addItem(QString::number(-1));
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

    cv::Mat img_result;
    // 画出所有框
    draw_rboxes_ids(this->img_1, img_result, this->contours_1);
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_1->size(),Qt::KeepAspectRatio);
    ui->labelImage_1->setPixmap(QPixmap::fromImage(dest));

    // 可选的框展示给用户
    ui->CB_roi_choose_4->clear();
    int num_boxes = this->contours_1.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_4->addItem(QString::number(i));
    }
    ui->CB_roi_choose_4->addItem(QString::number(-1));
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

    cv::Mat img_result;
    // 画出所有框
    draw_rboxes_ids(this->img_2, img_result, this->contours_2);
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_2->size(),Qt::KeepAspectRatio);
    ui->labelImage_2->setPixmap(QPixmap::fromImage(dest));

    // 可选的框展示给用户
    ui->CB_roi_choose_5->clear();
    int num_boxes = this->contours_2.size();
    for (int i=0; i<num_boxes; i++)
    {
        ui->CB_roi_choose_5->addItem(QString::number(i));
    }
    ui->CB_roi_choose_5->addItem(QString::number(-1));
    ui->CB_roi_choose_5->setCurrentIndex(num_boxes); //显示-1

}

void Probability::choose_roi(const QString &text)
{
    cv::Mat img_result;
    if (text.toInt() != -1)
    {
        int box_id = text.toInt();
        std::vector<cv::Point> contour = this->contours_1[box_id];
        // 存储选中的边界框
        this->contour_1.clear();
        this->contour_1.assign(contour.begin(), contour.end());

        // 将选中的框画在图像上
        std::vector<std::vector<cv::Point>> contour_;
        contour_.push_back(contour);

        // 画出选中的roi
        draw_rboxes(this->img_1, img_result, contour_);

    }
    else
    {
        draw_rboxes_ids(this->img_1, img_result, this->contours_1);
    }
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_1->size(),Qt::KeepAspectRatio);
    ui->labelImage_1->setPixmap(QPixmap::fromImage(dest));

}

void Probability::choose_roi_1(const QString &text)
{
    cv::Mat img_result;
    if (text.toInt() != -1)
    {
        this->choose_roi(text);

        // 选择背景区域
        float bg_ratio = ui->le_bg_ratio->text().toFloat();
        cv::RotatedRect fg_rbox;
        points2rbox(this->contour_1, fg_rbox);

        cv::Point2f center = fg_rbox.center;
        // 单位是角度
        float angle = fg_rbox.angle;
        float w = fg_rbox.size.width;
        float h = fg_rbox.size.height;

        // 背景区域
        float padding = bg_ratio * w / 2;
        float bg_w = w+2*padding;
        float bg_h = h+2*padding;
        cv::RotatedRect bg_rbox = cv::RotatedRect(center, cv::Size2f(bg_w,bg_h), angle);

        this->contour_2.clear();
        rbox2points(this->contour_2, bg_rbox);


        // 把前景框和背景框都画上去
        std::vector<std::vector<cv::Point>> contour_;
        contour_.push_back(this->contour_1);

        // 画出前景框
        draw_rboxes(this->img_1, img_result, contour_);

        contour_.clear();
        contour_.push_back(this->contour_2);
        // 画出背景框
        cv::Scalar color = cv::Scalar(0,255,0);
        draw_rboxes(img_result.clone(), img_result, contour_, -1, color);

    }
    else
    {
        draw_rboxes_ids(this->img_1, img_result, this->contours_1);
    }
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_1->size(),Qt::KeepAspectRatio);
    ui->labelImage_1->setPixmap(QPixmap::fromImage(dest));


}

void Probability::choose_roi_2(const QString &text)
{
    this->choose_roi(text);
}
void Probability::choose_roi_3(const QString &text)
{
    cv::Mat img_result;
    if (text.toInt() != -1)
    {
        int box_id = text.toInt();
        std::vector<cv::Point> contour = this->contours_1[box_id];
        // 存储选中的边界框
        this->contour_2.clear();
        this->contour_2.assign(contour.begin(), contour.end());


        // 把前景框和背景框都画上去
        std::vector<std::vector<cv::Point>> contour_;
        contour_.push_back(this->contour_1);

        // 画出前景框
        draw_rboxes(this->img_1, img_result, contour_);

        contour_.clear();
        contour_.push_back(this->contour_2);
        // 画出背景框
        cv::Scalar color = cv::Scalar(0,255,0);
        draw_rboxes(img_result.clone(), img_result, contour_, -1, color);

    }
    else
    {
        draw_rboxes_ids(this->img_1, img_result, this->contours_1);
    }
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_1->size(),Qt::KeepAspectRatio);
    ui->labelImage_1->setPixmap(QPixmap::fromImage(dest));

}

void Probability::choose_roi_4(const QString &text)
{
    this->choose_roi(text);
}
void Probability::choose_roi_5(const QString &text)
{
    cv::Mat img_result;
    if (text.toInt() != -1)
    {
        int box_id = text.toInt();
        std::vector<cv::Point> contour = this->contours_2[box_id];
        // 存储选中的边界框
        this->contour_2.clear();
        this->contour_2.assign(contour.begin(), contour.end());

        // 将选中的框画在图像上
        std::vector<std::vector<cv::Point>> contour_;
        contour_.push_back(contour);
        // 画出背景框
        cv::Scalar color = cv::Scalar(0,255,0);
        // 画出选中的roi
        draw_rboxes(this->img_2, img_result, contour_, -1, color);

    }
    else
    {
        draw_rboxes_ids(this->img_2, img_result, this->contours_2);
    }
    //显示图像
    QImage srcimg = MatToImage(img_result);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg.scaled(ui->labelImage_2->size(),Qt::KeepAspectRatio);
    ui->labelImage_2->setPixmap(QPixmap::fromImage(dest));
}

void Probability::show_CB_weights(const QString &text)
{
    if (text == QString("自定义法")) ui->GB_weights_inputs->show();
    else ui->GB_weights_inputs->hide();

}


void Probability::browse_save()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    // qDebug() << path;
    ui->le_savepath->setText(path);
}


void Probability::extract_fe()
{

    cv::Mat img = this->img_1.clone();
    // 选择特征
    if (this->fe_status.deep) extract_fe_deep(img, this->contours_1);
    if (this->fe_status.gray) extract_fe_gray(img, this->contours_1);
    if (this->fe_status.text) extract_fe_texture(img, this->contours_1);

}

// 根据旋转框的尺寸，映射到对应的特征层级
//at::Tensor map_roi_levels(at::Tensor rois, int num_levels, int finest_scale=56)
//{
//    /*
//     * Map rrois to corresponding feature levels by scales.
//        - scale < finest_scale: level 0
//        - finest_scale <= scale < finest_scale * 2: level 1
//        - finest_scale * 2 <= scale < finest_scale * 4: level 2
//        - scale >= finest_scale * 4: level 3

//        Args:
//            rois (Tensor): Input RRoIs, shape (k, 6). (index, x, y, w, h, angle)
//            num_levels (int): Total level number.

//         Returns:
//             Tensor: Level index (0-based) of each RoI, shape (k, )
//      */
//    at::Tensor scale = torch::sqrt(rois.index_select(1, torch::tensor({3})) * rois.index_select(1, torch::tensor({4})));
//    at::Tensor target_lvls = torch::floor(torch::log2(scale / finest_scale + 1e-6));
//    target_lvls = target_lvls.clamp(0, num_levels - 1).to(torch::kLong);

//    return target_lvls;
//}
at::Tensor map_roi_levels(at::Tensor rois, int num_levels, int finest_scale=56)
{
    /*
     * Map rrois to corresponding feature levels by scales.
        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RRoIs, shape (k, 6). (index, x, y, w, h, angle)
            num_levels (int): Total level number.

         Returns:
             Tensor: Level index (0-based) of each RoI, shape (k, )
      */
    at::Tensor scale = torch::sqrt(rois.index_select(1, torch::tensor({3})) * rois.index_select(1, torch::tensor({4})));
    at::Tensor target_lvls = torch::floor(torch::log2(scale / finest_scale + 1e-6));
    target_lvls = target_lvls.clamp(0, num_levels - 1).to(torch::kLong);

    return target_lvls;
}
void Probability::extract_fe_deep(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{
    std::string model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/extract_fe/exp361_no_align_extract_fe_script.pt";

    torch::Device device(torch::kCPU);

    at::Tensor input_tensor;
    cv::Mat img2 = img.clone();
    preprocess(img2, input_tensor);

    input_tensor = input_tensor.to(device);
//     cout << "input img size:" << input_tensor.sizes() << endl;


    torch::NoGradGuard no_grad;
    torch::jit::script::Module model;
    model = torch::jit::load(model_path, device);

//    cout << "导入模型完成"<<endl;
    ui->text_log->append("开始推理...");
    c10::intrusive_ptr<at::ivalue::Tuple> output;
    output = model.forward({input_tensor}).toTuple();

    // 先使用第一个特征图，以后再根据边界框的不同使用不同层级的特征图
    at::Tensor feat = output->elements()[0].toTensor();

    at::Tensor rois_1;
    points2xywhtheta(this->contour_1, rois_1);
    // 只有一张图像
    float batch_id =0.0;
    rois_1 = torch::cat({torch::tensor({batch_id}), rois_1}, 0);
    // shape [N, 6(batch_id, x(像素单位), y, w, h, theta(弧度))]
    rois_1 = rois_1.unsqueeze(0);

//    at::Tensor rois = torch::tensor(
//                {
//                    {0.0, 427.9353,616.8455, 119.1755,14.5517, -0.3343},
//                    {0.0, 60.4593, 156.7023, 186.1304, 22.0563, 1.5757}
//                }
//                ).to(device);

    int out_h=7, out_w=7, sample_num=2;
    int finest_scale=56*2; //Scale threshold of mapping to level 0.
    int featmap_strides[3] = {8,16,32};

    int num_levels=3;
    at::Tensor target_lvls = map_roi_levels(rois_1, num_levels, finest_scale=finest_scale);

    float spatial_scale = 1/8;
    int chn = feat.sizes()[1];
    int num_rois_1 = rois_1.sizes()[0];
    at::Tensor outs_rois_1 = torch::zeros({num_rois_1, chn, out_h, out_w}).to(device);
    roi_align_rotated_forward_cpu(feat, rois_1, out_h, out_w, spatial_scale, sample_num, outs_rois_1);
    outs_rois_1 = outs_rois_1.reshape(-1);





    at::Tensor rois_2;
    points2xywhtheta(this->contour_2, rois_2);
    // 只有一张图像
    rois_2 = torch::cat({torch::tensor({batch_id}), rois_2}, 0);
    // shape [N, 6(batch_id, x(像素单位), y, w, h, theta(弧度))]
    rois_2 = rois_2.unsqueeze(0);
    int num_rois_2 = rois_2.sizes()[0];
    at::Tensor outs_rois_2 = torch::zeros({num_rois_2, chn, out_h, out_w}).to(device);
    roi_align_rotated_forward_cpu(feat, rois_2, out_h, out_w, spatial_scale, sample_num, outs_rois_2);
    outs_rois_2 = outs_rois_2.reshape(-1);
//    outs_rois_2 = outs_rois_2 - outs_rois_1;

    outs_rois_1 /= outs_rois_1.norm(2);
    outs_rois_2 /= outs_rois_2.norm(2);
    this->roi_fe_1.deep = outs_rois_1.clone();
    this->roi_fe_2.deep = outs_rois_2.clone();


    ui->text_log->append("获得深度学习特征！");

}

void Probability::extract_fe_gray(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{

    // 根据4个点，找最小旋转矩形
    cv::RotatedRect rbox_1;
    points2rbox(this->contour_1, rbox_1);
    cv::Point2f center = rbox_1.center;
    // 单位是角度
    float angle = rbox_1.angle;
    // 前景区域
    int x = (int)(center.x);
    int y = (int)(center.y);
    int w = (int)(rbox_1.size.width);
    int h = (int)(rbox_1.size.height);
    int x1 = x - w/2;
    int y1 = y - h/2;
    // 左上角点坐标，w,h
    cv::Rect rect_roi_1(x1,y1,w,h);

    // 旋转图像
    cv::Mat img_rotate;
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, img_rotate, M, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 0);

    cv::Mat img_roi_1 = img_rotate(rect_roi_1);

//    QString crop_path = save_path.replace(QRegExp("."+fileSuffix), "_crop."+fileSuffix);
//    cv::imwrite((crop_path).toStdString(), roi);

    // 转化为灰度图
//    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);
//    std::vector<Mat>bgr_plane;
//    cv::split(roi, bgr_plane);
    cv::Mat roi_hist_1;
    get_hist(img_roi_1, roi_hist_1);


    // 提取第二个区域的特征
    cv::RotatedRect rbox_2;
    points2rbox(this->contour_2, rbox_2);
    center = rbox_2.center;
    // 单位是角度
    angle = rbox_2.angle;
    x = (int)(center.x);
    y = (int)(center.y);
    w = (int)(rbox_1.size.width);
    h = (int)(rbox_1.size.height);
    x1 = x - w/2;
    y1 = y - h/2;
    // 左上角点坐标，w,h
    cv::Rect rect_roi_2(x1,y1,w,h);

    // 旋转图像
    M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, img_rotate, M, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 0);
    cv::Mat img_roi_2 = img_rotate(rect_roi_2);
    cv::Mat roi_hist_2;
    get_hist(img_roi_2, roi_hist_2);
//    roi_hist_2 = roi_hist_2 - roi_hist_1;

//    double roimaxVal=0, bgmaxVal=0;
//    minMaxLoc(roi_hist, 0, &roimaxVal, 0, 0);
//    minMaxLoc(bg_hist, 0, &bgmaxVal, 0, 0);

//    roi_hist = roi_hist / roimaxVal;
//    bg_hist = bg_hist / bgmaxVal;
    at::Tensor roi_1_hist_tensor = torch::from_blob(roi_hist_1.data, {roi_hist_1.rows, roi_hist_1.cols, roi_hist_1.channels()});
    at::Tensor roi_2_hist_tensor = torch::from_blob(roi_hist_2.data, {roi_hist_2.rows, roi_hist_2.cols, roi_hist_2.channels()});

    roi_1_hist_tensor = roi_1_hist_tensor.reshape(-1);
    roi_2_hist_tensor = roi_2_hist_tensor.reshape(-1);


    roi_1_hist_tensor /= roi_1_hist_tensor.sum();
    roi_2_hist_tensor /= roi_2_hist_tensor.sum();

//    std::cout << roi_1_hist_tensor.min() << ":" << roi_1_hist_tensor.max() << std::endl;
//    std::cout << roi_2_hist_tensor.min() << ":" << roi_2_hist_tensor.max() << std::endl;

    this->roi_fe_1.gray = roi_1_hist_tensor;
    this->roi_fe_2.gray = roi_2_hist_tensor;

    ui->text_log->append("获得灰度特征！");

}

void Probability::extract_fe_texture(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{

    // 根据4个点，找最小旋转矩形
    cv::RotatedRect rbox_1;
    points2rbox(this->contour_1, rbox_1);
    cv::Point2f center = rbox_1.center;
    // 单位是角度
    float angle = rbox_1.angle;
    // 前景区域
    int x = (int)(center.x);
    int y = (int)(center.y);
    int w = (int)(rbox_1.size.width);
    int h = (int)(rbox_1.size.height);
    int x1 = x - w/2;
    int y1 = y - h/2;
    // 左上角点坐标，w,h
    cv::Rect rect_roi_1(x1,y1,w,h);

    // 旋转图像
    cv::Mat img_rotate;
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, img_rotate, M, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 0);
    cv::cvtColor(img_rotate, img_rotate, cv::COLOR_BGR2GRAY);


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
    std::vector<cv::Mat> imgs_filtered_1;
    for(int i = 0; i<num_theta; i++)
    {
        cv::Mat kernel1;
        cv::Mat dest;
        kernel1 = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta[i], lambd, gamma, psi, CV_32F);
        filter2D(img_rotate, dest, CV_32F, kernel1);
        imgs_filtered_1.push_back(dest);

    }
    // 合并为一个多通道图像
    cv::Mat mc_img_1;
    cv::merge(imgs_filtered_1, mc_img_1);


    cv::Mat roi_1_fe_text = mc_img_1(rect_roi_1);
    at::Tensor roi_1_tensor = torch::from_blob(
                roi_1_fe_text.data, {roi_1_fe_text.rows, roi_1_fe_text.cols, roi_1_fe_text.channels()});
    // 每个通道取均值
    at::Tensor roi_1_tensor_mean = roi_1_tensor.mean(0).mean(0);



    // 提取第二个区域的特征
    cv::RotatedRect rbox_2;
    points2rbox(this->contour_2, rbox_2);
    center = rbox_2.center;
    // 单位是角度
    angle = rbox_2.angle;
    x = (int)(center.x);
    y = (int)(center.y);
    w = (int)(rbox_1.size.width);
    h = (int)(rbox_1.size.height);
    x1 = x - w/2;
    y1 = y - h/2;
    // 左上角点坐标，w,h
    cv::Rect rect_roi_2(x1,y1,w,h);

    // 旋转图像
    M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, img_rotate, M, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 0);
    cv::cvtColor(img_rotate, img_rotate, cv::COLOR_BGR2GRAY);


    // gabor 纹理检测器，可以更多，
    std::vector<cv::Mat> imgs_filtered_2;
    for(int i = 0; i<num_theta; i++)
    {
        cv::Mat kernel1;
        cv::Mat dest;
        kernel1 = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta[i], lambd, gamma, psi, CV_32F);
        filter2D(img_rotate, dest, CV_32F, kernel1);
        imgs_filtered_2.push_back(dest);

    }
    // 合并为一个多通道图像
    cv::Mat mc_img_2;
    cv::merge(imgs_filtered_2, mc_img_2);


    cv::Mat roi_2_fe_text = mc_img_2(rect_roi_1);
    at::Tensor roi_2_tensor = torch::from_blob(
                roi_2_fe_text.data, {roi_2_fe_text.rows, roi_2_fe_text.cols, roi_2_fe_text.channels()});
    at::Tensor roi_2_tensor_mean = roi_2_tensor.mean(0).mean(0);
//    roi_2_tensor_mean = roi_2_tensor_mean - roi_1_tensor_mean;


    // 归一化
    roi_1_tensor_mean /= roi_1_tensor_mean.norm(2);
    roi_2_tensor_mean /= roi_2_tensor_mean.norm(2);

    this->roi_fe_1.text = roi_1_tensor_mean;
    this->roi_fe_2.text = roi_2_tensor_mean;


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
        calcHist( &img, 1, channels, cv::Mat(), // do not use mask
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

    if (ui->CB_weights->currentText() == QString("自定义法"))
    {
        this->fe_similarity_weights = {1.0/3.0, 1.0/3.0, 1.0/3.0};
        this->fe_similarity_weights.deep = ui->le_deep_weight->text().toFloat();
        this->fe_similarity_weights.gray = ui->le_gray_weight->text().toFloat();
        this->fe_similarity_weights.text = ui->le_text_weight->text().toFloat();
    }
    else if (ui->CB_weights->currentText() == QString("熵权法"))
    {
        this->fe_similarity_weights.deep = 0.5;
        this->fe_similarity_weights.gray = 0.3;
        this->fe_similarity_weights.text = 0.2;
    }

    float si = 0;
    if (this->fe_status.deep) si = si + this->fe_similarity.deep * this->fe_similarity_weights.deep;
    if (this->fe_status.gray) si = si + this->fe_similarity.gray * this->fe_similarity_weights.gray;
    if (this->fe_status.text) si = si + this->fe_similarity.text * this->fe_similarity_weights.text;

    this->similarity = si;

}

// 巴氏距离
float distance_B(at::Tensor p, at::Tensor q)
{
    // 巴氏系数
    at::Tensor B = torch::sqrt(p*q).sum();
    // 巴氏距离
    float a = torch::log(B).item().toFloat();

}
void Probability::cal_similarity_deep()
{

//    cout << "roi_fe:" <<this->roi_fe.deep << endl;
    at::Tensor diff = torch::abs(this->roi_fe_1.deep - this->roi_fe_2.deep);
    float similarity= 1 - diff.mean().item().toFloat();
    this->fe_similarity.gray = similarity;

    QString log = QString("\n深度学习特征相似度为:") + QString::number(similarity, 'f', 3);
    ui->text_log->append(log);
}


void Probability::cal_similarity_gray()
{
    // 灰度直方图，使用巴氏系数作为特征相似度
    float similarity = torch::sqrt(this->roi_fe_1.gray*this->roi_fe_2.gray).sum().item().toFloat();

    this->fe_similarity.gray = similarity;

    QString log = QString("\n灰度特征相似度为:") + QString::number(similarity, 'f', 3);
    ui->text_log->append(log);
}


void Probability::cal_similarity_text()
{
    at::Tensor diff = torch::abs(this->roi_fe_1.text - this->roi_fe_2.text);

    float similarity= 1 - diff.mean().item().toFloat();
    this->fe_similarity.text = similarity;
    QString log = QString("\n纹理特征相似度为:") + QString::number(similarity, 'f', 3);
    ui->text_log->append(log);

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
    QString map_type = ui->CB_map->currentText();

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

    ui->le_probability->setText(QString::number(this->probability));
    QString log = QString("\n计算得到的识别概率为:") + QString::number(this->probability, 'f', 3);
    ui->text_log->append(log);
}

void Probability::save_results()
{
    QString path = ui->le_savepath->text();
    QString img_path, img_path2;
    if (this->rois_type == QString("目标-背景")) img_path = ui->le_imgpath_1->text();
    if (this->rois_type == QString("目标-目标")) img_path = ui->le_imgpath_2->text();
    if (this->rois_type == QString("目标-目标2"))
    {
        img_path = ui->le_imgpath_3->text();
        QString img_path2 = ui->le_imgpath_3->text();
    }

    QFileInfo imginfo = QFileInfo(img_path);
    QString img_name = imginfo.fileName();
    QString fileSuffix = imginfo.suffix();

    // 保存检测结果
    QString txt_name = img_name;
    txt_name.replace(QRegExp("."+fileSuffix), QString("_probability.txt"));
    QString txt_save_path = path +"/"+ txt_name;

    std::ofstream fout;
    fout.open(txt_save_path.toStdString());

    fout << this->rois_type.toStdString() + "\n";
    fout << img_path.toStdString() + "\n";
    if (this->rois_type == QString("目标-目标2")) fout << img_path2.toStdString() + "\n";

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

void Probability::on_bu_rois_group(QAbstractButton *button)
{
    this->rois_type = button->text();
    this->reset_show();

}

void reset_QCB(QComboBox *CB_box)
{
    // 如果QComboBox只有一个-1，此时clear会导致text发生变化，从而导致信号触发，因此在clear前先断开信号连接
    CB_box->blockSignals(true);//true屏蔽信号
    CB_box->clear();
    CB_box->addItem(QString::number(-1));
    CB_box->blockSignals(false);//false取消屏蔽信号
}
void Probability::reset_show()
{
    ui->le_imgpath_1->clear();
    ui->le_imgpath_2->clear();
    ui->le_imgpath_3->clear();
    ui->le_imgpath_4->clear();

    ui->le_gtpath_1->clear();
    ui->le_gtpath_2->clear();
    ui->le_gtpath_3->clear();
    ui->le_gtpath_4->clear();

    // 如果QComboBox只有一个-1，此时clear会导致text发生变化，从而导致信号触发，因此在clear前先断开信号连接
    reset_QCB(ui->CB_roi_choose_1);
    reset_QCB(ui->CB_roi_choose_2);
    reset_QCB(ui->CB_roi_choose_3);
    reset_QCB(ui->CB_roi_choose_4);
    reset_QCB(ui->CB_roi_choose_5);

    ui->labelImage_1->clear();
    ui->labelImage_2->clear();

}
