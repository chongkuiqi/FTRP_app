#ifndef SARDEDUCTION_H
#define SARDEDUCTION_H

#include <QMainWindow>
#include <QFileDialog>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
//using namespace cv;
namespace Ui {
class SARdeduction;
}

class SARdeduction : public QMainWindow
{
    Q_OBJECT

public:
    explicit SARdeduction(QWidget *parent = nullptr);
    ~SARdeduction();
signals:
    void sarexit();


private slots:
    void on_sarexit_clicked();

    void on_browse_clicked();

    void on_run_clicked();

    void on_save_clicked();

private:
    Ui::SARdeduction *ui;
};

#endif // SARDEDUCTION_H
