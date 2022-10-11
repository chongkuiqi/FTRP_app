#include "sardeduction.h"
#include "ui_sardeduction.h"
#include <QtDebug>
#include <QDir>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <QCompleter>
#include <QStringList>

using namespace cv;
using std::string;
//using namespace std;
SARdeduction::SARdeduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SARdeduction)
{
    ui->setupUi(this);
    QStringList azi_list;
    azi_list<<"1.8";
    QStringList ran_list;
    ran_list<<"2";
    QCompleter *com1 = new QCompleter(azi_list,this);
    ui->Azimuth->setCompleter(com1);
    QCompleter *com2 = new QCompleter(ran_list,this);
    ui->Range->setCompleter(com2);


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
void Add_RayleighNoise(Mat img_input,Mat& img_output,double mean, double sigma,int k)
{
    img_output.create(img_input.rows,img_input.cols,img_input.type());
    for (int x =0;x<img_input.rows;x++)
    {
        for (int y = 0;y<img_input.cols;y++)
        {
            double temp = saturate_cast<uchar>(img_input.at<uchar>(x,y)-k*generateGaussianNoise(mean,sigma));
            img_output.at<uchar>(x,y)=temp;
        }
    }

}

//获得保存路径
void string_replace(string &s1, const string &s2,const string &s3)
{
    string::size_type pos = 0;
    string::size_type a = s2.size();
    string::size_type b = s3.size();
    while((pos=s1.find(s2,pos))!=string::npos)
    {
        s1.replace(pos,a,s3);
        pos+=b;
    }
}

void SARdeduction::on_run_clicked()
{
    QString Azimuth = ui->Azimuth->text();
    QString Range = ui->Range->text();
    QString path = ui->fileline->text();
    //获取图像名称
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();


    //读图像
    Mat image = cv::imread(path.toStdString());
    //灰度化

    Mat image_gray;
    cvtColor(image,image_gray,COLOR_BGR2GRAY);
    float azimuth = Azimuth.toFloat();
    float range = Range.toFloat();
    float b = 4/(azimuth*range);
    image_gray = image_gray*b;

    //添加噪声
    Mat image_output;
    Add_RayleighNoise(image_gray,image_output,0,30,1);

    //保存
    QString save_folder = ui->save_path_line->text();
    string save_path = save_folder.toStdString();
    string finalpath = save_path+"/"+file_name.toStdString();
    std::cout<<finalpath<<std::endl;
    cv::imwrite(finalpath,image_output);
    qDebug()<<"finish";

}

void SARdeduction::on_save_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}

