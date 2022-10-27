#ifndef PROBABILITY_H
#define PROBABILITY_H


#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include <QButtonGroup>

#undef slots
#include "torch/torch.h"
#include <torch/script.h>
#define slots Q_SLOTS

#include "util.h"

// 泛型编程，存储各个特征的状态、特征相似度
template <typename T>
struct struct_fe
{
    T deep;
    T gray;
    T text;
};


namespace Ui {
class Probability;
}

class Probability : public QMainWindow
{
    Q_OBJECT

public:
    explicit Probability(QWidget *parent = nullptr);
    ~Probability();

    void browse_img(QString type);
    void browse_img_1();
    void browse_img_2();
    void browse_img_3();
    void browse_img_4();

    void browse_gt_1();
    void browse_gt_2();
    void browse_gt_3();
    void browse_gt_4();

    // roi区域选择
    void choose_roi(const QString &text);
    void choose_roi_1(const QString &text);
    void choose_roi_2(const QString &text);
    void choose_roi_3(const QString &text);
    void choose_roi_4(const QString &text);
    void choose_roi_5(const QString &text);

    // 是否显示特征相似度加权权值输入界面
    void show_CB_weights(const QString &text);

    void browse_save();

    void extract_fe();

    void extract_fe_deep(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours);
    void extract_fe_gray(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours);
    void extract_fe_texture(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours);
    void get_hist(cv::Mat & img, cv::Mat & hist);

    void cal_similarity();
    void cal_similarity_deep();
    void cal_similarity_gray();
    void cal_similarity_text();



    void cal_probability();

    void save_results();

    void on_bu_fe_group(QAbstractButton *button);


    // 特征提取区域选择
    void on_bu_rois_group(QAbstractButton *button);

    // 显示复位，各个按钮、图像显示全部清零
    void reset_show();


    void change_bg_ratio(const QString text);


private:
    Ui::Probability *ui;

    cv::Mat img_1;
    cv::Mat img_2;

    float similarity;
    float probability;

    // 待提取特征区域选择按钮组
    QButtonGroup bu_rois_group;
    // 待提取特征区域类型，包括3种，“目标-背景”，“目标-目标”，“目标-目标2”
    QString rois_type = QString("目标-背景");


    // 特征选择按钮组
    QButtonGroup bu_fe_group;

    // 两个contours
    std::vector<std::vector<cv::Point>> contours_1;
    std::vector<std::vector<cv::Point>> contours_2;

    std::vector<cv::Point> contour_1;
    std::vector<cv::Point> contour_2;

//    // 两个待提取特征区域的旋转框
//    cv::RotatedRect rrect_box_1;
//    cv::RotatedRect rrect_box_2;


    // 旋转后的图像
    cv::Mat img_rotate;


    // 存储各个特征的状态，即是否提取该特征
    struct_fe<bool> fe_status = {false,false, false};

    // 前景/背景区域的特征
    struct_fe<at::Tensor> roi_fe_1;
    struct_fe<at::Tensor> roi_fe_2;

    // 存储各个特征的相似度
    struct_fe<float> fe_similarity = {0.0, 0.0, 0.0};

    // 存储各个特征的相似度加权权值
    struct_fe<float> fe_similarity_weights = {1.0/3.0, 1.0/3.0, 1.0/3.0};



};

#endif // PROBABILITY_H
