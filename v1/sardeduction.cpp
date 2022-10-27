#include "sardeduction.h"
#include "ui_sardeduction.h"
#include <QtDebug>
#include <QDir>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <QStringList>
#include <QLabel>

//#include <string>
//using namespace cv;
//using std::string;
//using namespace std;
SARdeduction::SARdeduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SARdeduction)
{
    ui->setupUi(this);



}

SARdeduction::~SARdeduction()
{
    delete ui;
}

void SARdeduction::on_sarexit_clicked()
{
    emit sarexit();
}


void SARdeduction::on_browse_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,"open","../","all(*.*)");
    ui->fileline->setText(path);
    //显示图像
    QImage* srcimg = new QImage;
    srcimg -> load(path);
    ui->src_image->setPixmap(QPixmap::fromImage(*srcimg).scaled(400,400));
}



//产生瑞利随机数
double generateGaussianNoise(double mean, double sigma)
{
    double v1,S,u1;
    srand((unsigned)time(NULL));
    u1 = (double)rand()/RAND_MAX;//均匀分布随机数
    v1 = -2*log(u1);
    S = sigma*sqrt(v1)+mean;
    return S;
}
//噪声图像
void Add_RayleighNoise(cv::Mat img_input,cv::Mat& img_output,double mean, double sigma,int k)
{
    img_output.create(img_input.rows,img_input.cols,img_input.type());
    for (int x =0;x<img_input.rows;x++)
    {
        for (int y = 0;y<img_input.cols;y++)
        {
            double temp = cv::saturate_cast<uchar>(img_input.at<uchar>(x,y)-k*generateGaussianNoise(mean,sigma));
            img_output.at<uchar>(x,y)=temp;
        }
    }

}


void SARdeduction::on_run_clicked()
{
//    QString Azimuth = ui->Azimuth->text();
    QString Azimuth = ui->Azimuth->currentText();
    QString Range = ui->Range->currentText();
    QString path = ui->fileline->text();
    QString spe_Azimuth = ui->spe_azi->currentText();
    QString spe_Range = ui->spe_ran->currentText();
    //获取图像名称
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();
    QString save_folder = ui->save_path_line->text();
    ui->log->clear();
    ui->log->append("参数读取完毕");

    //读图像

    cv::Mat image = cv::imread(path.toStdString());
//    namedWindow("Display window",WINDOW_AUTOSIZE);
//    imshow("Display window",image);
//    waitKey(1);
    cv::Mat image_gray;
    cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);
    float azimuth = Azimuth.toFloat();
    float range = Range.toFloat();
    float spe_azimuth = spe_Azimuth.toFloat();
    float spe_range = spe_Range.toFloat();
    float b = (spe_azimuth*spe_range)/(azimuth*range);

    image_gray = image_gray*b;
    //添加噪声
    cv::Mat image_output;
    Add_RayleighNoise(image_gray,image_output,0,30,1);
    //保存
    std::string save_path = save_folder.toStdString();
    std::string finalpath = save_path+"/"+file_name.toStdString();
    cv::imwrite(finalpath,image_output);
    //显示输出图像
    QImage* outimg = new QImage;
    outimg -> load(QString::fromStdString(finalpath));
    ui->output_img->setPixmap(QPixmap::fromImage(*outimg).scaled(400,400));
    ui->log->append("推演完成，图像保存至：");
    ui->log->append(QString::fromStdString(finalpath));

}

void SARdeduction::on_save_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}

