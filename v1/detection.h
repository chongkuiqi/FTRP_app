#ifndef DETECTION_H
#define DETECTION_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

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

    void detect();

    void save_results();

private:
    Ui::Detection *ui;

    // 保存图像
    cv::Mat img;
    std::vector<std::vector<cv::Point>> contours;

    cv::Mat img_result;

};

#endif // DETECTION_H
