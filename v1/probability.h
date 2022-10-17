#ifndef PROBABILITY_H
#define PROBABILITY_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include <QButtonGroup>

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

    void browse_img();
    void browse_gt();

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

private:
    Ui::Probability *ui;

    // 前景区域的特征和背景区域的特征
    cv::Mat roi_fe_deep;
    cv::Mat bg_fe_deep;
    // 前景区域的特征和背景区域的特征
    cv::Mat roi_fe_gray;
    cv::Mat bg_fe_gray;
    // 前景区域的特征和背景区域的特征
    cv::Mat roi_fe_text;
    cv::Mat bg_fe_text;

    float similarity;

    float probability;

    // 特征选择额按钮组
    QButtonGroup bu_fe_group;

    // 存储各个特征的状态，即是否提取该特征
    struct_fe<bool> fe_status = {false,false, false};

    // 前景/背景区域的特征
    struct_fe<cv::Mat> roi_fe;
    struct_fe<cv::Mat> bg_fe;

    // 存储各个特征的相似度
    struct_fe<float> fe_similarity = {0.0, 0.0, 0.0};

};

#endif // PROBABILITY_H
