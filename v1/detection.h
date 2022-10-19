#ifndef DETECTION_H
#define DETECTION_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>


// 泛型编程，存储各个特征的状态、特征相似度
template <typename T>
struct struct_MS
{
    T Opt;
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

    void browse_img();
    void browse_model();
    void browse_save();

    void detect();

    void save_results();

private:
    Ui::Detection *ui;

    // 保存图像
    cv::Mat img;
    std::vector<std::vector<cv::Point>> contours;

    // 图像可视化结果
    cv::Mat img_result;

};

#endif // DETECTION_H
