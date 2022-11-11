#ifndef OPTDEDUCTION_H
#define OPTDEDUCTION_H

#include <QMainWindow>
#include <QFileDialog>

//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <QString>
#define IMG_SIZE 1024
#define IMG_CHN 3
namespace Ui {
class optdeduction;
}

class optdeduction : public QMainWindow
{
    Q_OBJECT

public:
    explicit optdeduction(QWidget *parent = nullptr);
    ~optdeduction();
signals:
    void optexit();

private slots:
    void on_exit_clicked();

    void on_browse_clicked();


    void on_classify_clicked();

    void on_save_clicked();





    void on_initialize_clicked();

private:
    void gen_input_txt(QString solar_zenith_angle,QString solar_azimuth,QString satellite_zenith_angle,QString satellite_aximuth\
                       ,int atmospheric_model,int type_o_aerosol\
                       ,QString altitude_of_target,int spectral_conditions);
    int select_DN(int label);
    void label2rgb(int label);
    void initialize();
    void get_DN(float gain,float bais);

    QStringList phase;
    QStringList ref;
    QStringList appar;
    QStringList bgr;
    int DN[7];
    int DN_norm[7];

    Ui::optdeduction *ui;

};

#endif // OPTDEDUCTION_H
