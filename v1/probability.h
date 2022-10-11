#ifndef PROBABILITY_H
#define PROBABILITY_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>



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
    void extract__fe_texture(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours);
    void get_hist(cv::Mat & img, cv::Mat & hist);

    void cal_similarity();
    void cal_probability();

    void save_results();

private:
    Ui::Probability *ui;

    cv::Mat roi_fe;
    cv::Mat bg_fe;

    float similarity;

    float probability;
};

#endif // PROBABILITY_H
