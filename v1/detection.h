#ifndef DETECTION_H
#define DETECTION_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include <QButtonGroup>

// 泛型编程，存储各个特征的状态、特征相似度
template <typename T>
struct struct_MS
{
    T opt;
    T IR;
    T SAR;
};


namespace Ui {
class Detection;
}

class Detection : public QMainWindow
{
    Q_OBJECT

public:
    explicit Detection(QWidget *parent = nullptr);
    ~Detection();

    void browse_img(QString img_type);
    void browse_img_opt();
    void browse_img_IR();
    void browse_img_SAR();

    void browse_save();
    void detect();

    void save_results();

    void switch_mode(const QString &text);
    void switch_mode_show(const QString &text);
    void on_bu_group_SS(QAbstractButton *button);

    void on_bu_group_MS(QAbstractButton *button);

    void get_bu_group_status(QButtonGroup *bu_group);

    void show_img_results(QString img_type);
    void show_img_opt_results();
    void show_img_IR_results();
    void show_img_SAR_results();

    void save_img(QString img_type);
    void save_img_opt();
    void save_img_IR();
    void save_img_SAR();



    void reset_show();

private:
    Ui::Detection *ui;

    // 保存图像
    cv::Mat img;
    std::vector<std::vector<cv::Point>> contours;

    // 图像可视化结果
    cv::Mat img_result;

    // 单频谱选择按钮组
    QButtonGroup bu_group_SS;
    // 多频谱选择按钮组
    QButtonGroup bu_group_MS;

    // 存储各个特征的状态，即是否提取该特征
    struct_MS<bool> img_status = {false,false, false};

};

#endif // DETECTION_H
