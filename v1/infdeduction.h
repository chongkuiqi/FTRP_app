#ifndef INFDEDUCTION_H
#define INFDEDUCTION_H

#include <QMainWindow>
#include <QFileDialog>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
namespace Ui {
class infdeduction;
}

class infdeduction : public QMainWindow
{
    Q_OBJECT

public:
    explicit infdeduction(QWidget *parent = nullptr);
    ~infdeduction();
signals:
    void infexit();

private slots:
    void on_browser_clicked();

    void on_browser_2_clicked();

    void on_run_clicked();

    void on_initialize_clicked();
    void initialize();

    void on_exit_clicked();

    void on_save_clicked();

private:
    Ui::infdeduction *ui;
    cv::Mat final;
};

#endif // INFDEDUCTION_H
