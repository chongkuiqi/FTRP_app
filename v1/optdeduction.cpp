#include "optdeduction.h"
#include "ui_optdeduction.h"
#include <QtDebug>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <cmath>
#include <QDir>
#include <QCompleter>
#include <QStringList>

using namespace cv;
using std::string;
optdeduction::optdeduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::optdeduction)
{
    ui->setupUi(this);
    QStringList air_list;
    air_list<<"0.26";
    QStringList plant_list;
    plant_list<<"0.07";
    QStringList road_list;
    road_list<<"0.09";
    QStringList land_list;
    land_list<<"0.15";
    QStringList gain_list;
    gain_list<<"0.1887";
    QStringList space_list;
    space_list<<"1";
    QStringList angle_list;
    angle_list<<"65.46";
    QCompleter *com1 = new QCompleter(air_list,this);
    ui->airres->setCompleter(com1);
    QCompleter *com2 = new QCompleter(plant_list,this);
    ui->plant->setCompleter(com2);
    QCompleter *com3 = new QCompleter(road_list,this);
    ui->road->setCompleter(com3);
    QCompleter *com4 = new QCompleter(land_list,this);
    ui->land->setCompleter(com4);
    QCompleter *com5 = new QCompleter(gain_list,this);
    ui->gain->setCompleter(com5);
    QCompleter *com6 = new QCompleter(space_list,this);
    ui->spaceres->setCompleter(com6);
    QCompleter *com7 = new QCompleter(angle_list,this);
    ui->angle->setCompleter(com7);
}

optdeduction::~optdeduction()
{
    delete ui;
}

void optdeduction::on_exit_clicked()
{
    emit optexit();
}


void optdeduction::on_browse_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,"open","../","all(*.*)");
    ui->fileline->setText(path);
}

void replace_path(string &s1, const string &s2,const string &s3)
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



float select(int label,float plant,float road,float land)
{

    switch(label)
    {
        case 0:
            return plant;
        case 1:
            return plant;
        case 2:
            return land;
        case 3:
            return road;
        case 4:
            return road;

    }

}

float cal_L(float ref,float theta)
{
    int Es = 1367;
    int d = 1;
    float L;
    float theta_h = theta/180*M_PI;
    L = ref*cos(theta_h)*Es/(M_PI*d*d);
    return L;

}
int get_DN(float L,float Gain)
{
    int Bias = 1;
    int DN = (L-Bias)/Gain;
    return DN;
}

void optdeduction::on_classify_clicked()
{
    QString airres = ui->airres->text();
    QString plant = ui->plant->text();
    QString road = ui->road->text();
    QString land = ui->land->text();
    QString angle = ui->angle->text();
    QString Gain = ui->gain->text();
    QString spaceres = ui->spaceres->text();
//    QString path = ui->fileline->text();
//    qDebug()<<airres<<plant<<road<<land<<angle<<Gain<<spaceres;
    float plant_ = plant.toFloat();
    float road_  = road.toFloat();
    float land_  = land.toFloat();
    float angle_ = angle.toFloat();
    float Gain_ = Gain.toFloat();
    float airres_ = airres.toFloat();
    float spaceres_ = spaceres.toFloat();

    //输入图像路径
    QString path = ui->fileline->text();
    //获取图像名称
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();
    //读取图像
    Mat src_image = cv::imread(path.toStdString());
    //转成灰度图
//    Mat image_gray;
    int width = src_image.cols;
    int height = src_image.rows;
    int channels = src_image.channels();

    int sampleCount = width*height;
    int clusterCount = 5;//分类数
    Mat points(sampleCount,channels,CV_32F,Scalar(10));//points用来保存所有的数据
    Mat labels;//聚类后的标签
    Mat center(clusterCount,1,points.type());//聚类后的类别中心
    //将图像的RGB像素转到样本数据
    int index;
    for (int i=0;i<src_image.rows;i++)
    {
        for(int j=0;j<src_image.cols;j++)
        {
            index = i*width+j;
            Vec3b bgr = src_image.at<Vec3b>(i,j);
            //将图像中的每个通道数据分别复制给points的值
            points.at<float>(index,0) = static_cast<int>(bgr[0]);
            points.at<float>(index,1) = static_cast<int>(bgr[1]);
            points.at<float>(index,2) = static_cast<int>(bgr[2]);
        }
    }
    //运行K-means算法
    TermCriteria criteria = TermCriteria(TermCriteria::EPS +TermCriteria::COUNT,10,0.1);
    kmeans(points,clusterCount,labels,criteria,3,KMEANS_PP_CENTERS,center);
    //显示图像分割结果

    Mat output = Mat::zeros(src_image.size(),src_image.type());
    for (int i = 0;i<src_image.rows;i++)
    {
        for (int j=0;j<src_image.cols;j++)
        {
            index = i*width+j;
            int label = labels.at<int>(index);//每一个像素属于哪个标签
            float ref = select(label,plant_,road_,land_);
            float L =  cal_L(ref,angle_);
            int DN = get_DN(L,Gain_);
            output.at<Vec3b>(i,j)[0]=DN;
            output.at<Vec3b>(i,j)[1]=DN;
            output.at<Vec3b>(i,j)[2]=DN;


        }
    }
    //resize
    int new_width = width*airres_/spaceres_;
    int new_height = height*airres_/spaceres_;

    Size dsize = Size(new_width,new_height);
    Mat final;
    cv::resize(output,final,dsize,0,0,INTER_LINEAR);
    //保存
    QString save_folder = ui->save_path_line->text();
    string save_path = save_folder.toStdString();
    string finalpath = save_path+"/"+file_name.toStdString();
    imwrite(finalpath,final);


}


void optdeduction::on_save_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}

