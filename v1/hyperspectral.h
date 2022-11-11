#ifndef HYPERSPECTRAL_H
#define HYPERSPECTRAL_H

#include <QMainWindow>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
using namespace std;

namespace Ui {
class hyperspectral;
}

class hyperspectral : public QMainWindow
{
    Q_OBJECT

public:
    explicit hyperspectral(QWidget *parent = nullptr);
    ~hyperspectral();

signals:
    void hypexit();
private slots:
    void on_browser_clicked();

    void on_preview_clicked();

    void on_srcSlider_sliderMoved(int position);

    void on_run_clicked();

    void on_browser_2_clicked();

    void on_initialize_clicked();

    void on_exit_clicked();

    void on_fileline_textChanged(const QString &arg1);

    void on_width_textChanged(const QString &arg1);

    void on_height_textChanged(const QString &arg1);

    void on_num_spec_textChanged(const QString &arg1);

private:
    Ui::hyperspectral *ui;
    vector<vector<double>> read_csv_file(string filename);
    vector<vector<double>> hyper_matrix;
    vector<cv::Mat> image_vec;
    vector<cv::Mat> out_image_vec;
    void initialize();
};

#endif // HYPERSPECTRAL_H
