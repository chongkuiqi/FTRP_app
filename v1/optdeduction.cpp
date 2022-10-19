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
#include <QProcess>
#include <QFile>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string>
using namespace cv;
using namespace std;
using std::string;

optdeduction::optdeduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::optdeduction)
{
    ui->setupUi(this);
    phase<<"vegetation"<<"clear_water"<<"dry_sand"<<"lake_water"<<"volcanic_debris"<<"road"<<"land";
    ref<<"0.09"<<"0.15";




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
    //显示图像
    QImage* srcimg = new QImage;
    srcimg -> load(path);
    ui->srcimg->setPixmap(QPixmap::fromImage(*srcimg));
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



float optdeduction::select_appar(int label)
{
    int id;
    switch(label)
    {
        case 0:
            id = phase.indexOf("vegetation");
//            qDebug()<<id;
            return appar.at(id).toFloat();
        case 1:
            id = phase.indexOf("vegetation");
//            qDebug()<<id;
            return appar.at(id).toFloat();
        case 2:
            id = phase.indexOf("land");
//            qDebug()<<id;
            return appar.at(id).toFloat();
        case 3:
            id = phase.indexOf("road");
//            qDebug()<<id;
            return appar.at(id).toFloat();
        case 4:
            id = phase.indexOf("road");
//            qDebug()<<id;
            return appar.at(id).toFloat();

    }

}


int get_DN(float L,float Gain,float Bias)
{
//    int Bias = 1;
    int DN = (L-Bias)/Gain;
    return DN;
}
void optdeduction::gen_input_txt(QString solar_zenith_angle,QString solar_azimuth,QString satellite_zenith_angle,QString satellite_aximuth,QString month\
                   ,QString date,int atmospheric_model,int type_o_aerosol,int concentration,QString v_value\
                   ,QString altitude_of_target,int spectral_conditions){
//    QStringList phase;
//    phase<<"vegetation"<<"clear_water"<<"dry_sand"<<"lake_water"<<"volcanic_debris";
    QFile sh("./6s.sh");
    sh.open(QFile::WriteOnly|QFile::Truncate);
    QTextStream sh_add(&sh);
    sh_add<<"#!/bin/bash\n";
    for(int i = 0;i<phase.size();i++){
        //修改sh文件
        QString line_1 = "wine 6s.exe<input_"+phase.at(i)+".txt>output_"+phase.at(i)+".txt\n";
        QString line_2 = "grep \"appar. rad.(w/m2/sr/mic)\" output_"+phase.at(i)+".txt>appar_"+phase.at(i)+".txt\n";
        sh_add<<line_1;
        sh_add<<line_2;

        //修改input文件
        QString name = "input_"+phase.at(i)+".txt";
        QFile input_txt(name);
        qDebug()<<name;
        input_txt.open(QFile::WriteOnly|QFile::Truncate);
        QTextStream out(&input_txt);
        //1 几何参数——用户自己选择观测几何参数
        out<<"0\n";
        //太阳天顶角，太阳方位角，卫星天顶角，卫星方位角，月，日
        out<<solar_zenith_angle<<" "<<solar_azimuth<<" "<<satellite_zenith_angle<<" "<<satellite_aximuth<<" "<<month<<" "<<date<<"\n";
        //2 大气模式
        out<<QString::number(atmospheric_model+1,10)<<"\n";
        //3 气溶胶类型
        out<<QString::number(type_o_aerosol+1,10)<<"\n";
        //4 气溶胶含量
        /*if (concentration==0){
            out<<"-1\n";
        }else */
        if(concentration==0){
            out<<"0\n"<<v_value<<"\n";
        }else if(concentration==1){
            out<<"1\n"<<v_value<<"\n";
        }
        //5 目标高度
        if (altitude_of_target=="目标在海平面高度"){
            out<<"0\n";
        }else{
            out<<"-"+altitude_of_target<<"\n";
        }
        //6 传感器高度参数
        out<<"-1000\n";
        //7 光谱参数；
        out<<QString::number(spectral_conditions+2,10)<<"\n";
        //8 地表反射率类型
        out<<"0\n"<<"0\n";//均匀表面,无方向效应
        //反射率
//        out<<"1\n";
        if (i<5){
            out<<QString::number(i+1,10)<<"\n";
        }else{
            out<<"0\n"<<ref.at(i-5)<<"\n";
        }

        //不激活大气校正方式
        out<<"-2\n";
        input_txt.close();
    }
    sh.close();






}


void optdeduction::on_classify_clicked()
{
    //参数读取
    //观测辐亮度计算参数
    QString solar_zenith_angle = ui->solar_zenith_angle->currentText();//太阳天顶角
    QString solar_azimuth = ui->solar_azimuth->currentText();//太阳方位角
    QString satellite_zenith_angle = ui->satellite_zenith_angle->currentText();//卫星天顶角
    QString satellite_aximuth = ui->satellite_aximuth->currentText();//卫星方位角
    QString month = ui->month->currentText();//月
    QString date = ui->date->currentText();//日
    int atmospheric_model = ui->atmospheric_model->currentIndex();//大气模式
    int type_o_aerosol = ui->type_o_aerosol->currentIndex();//气溶胶模式
    int concentration = ui->concentration->currentIndex();//气溶胶含量
    qDebug()<<"气溶胶含量"<<concentration;
    QString v_value = ui->v_value->currentText();//气溶胶含量
    qDebug()<<"value"<<v_value;
    QString altitude_target = ui->altitude_target->currentText();//目标高度
    int spectral_conditions = ui->spectral_conditions->currentIndex();//光谱参数

    //辐射定标计算参数
    QString Gain = ui->gain->currentText();
    QString Bias = ui->bias->currentText();
    float Gain_ = Gain.toFloat();
    float Bias_ = Bias.toFloat();
    //分辨率转换参数
    QString airres = ui->airres->currentText();
    QString spaceres = ui->spaceres->currentText();
    float airres_ = airres.toFloat();
    float spaceres_ = spaceres.toFloat();
    ui->log->clear();
    ui->log->append("参数读取完毕");
    qDebug()<<"参数读取完毕";

    //生成input.txt
    gen_input_txt(solar_zenith_angle,solar_azimuth,satellite_zenith_angle,satellite_aximuth,month\
                  ,date,atmospheric_model,type_o_aerosol,concentration,v_value\
                  ,altitude_target,spectral_conditions);
    ui->log->append("6s计算模型输入文件生成完毕");
    //调用6S.exe
    qDebug()<<"调用6S";
    QProcess myProcess;
    myProcess.execute("./6s.sh");
    ui->log->append("表观辐亮度计算完毕");
    //读取各表观辐亮度
    for (int pha=0;pha<phase.size() ;++pha ) {
        QString apparfile_name = "./appar_"+phase.at(pha)+".txt";
        QFile appar_file(apparfile_name);
        if(!appar_file.open(QIODevice::ReadOnly|QIODevice::Text))
        {
            qDebug()<<"can't open the file!";
        }
        QByteArray line = appar_file.readLine();
        QString str(line);
        QStringList list = str.split("  ");
        appar.append(list[8]);
//        qDebug()<<list[8];
        ui->log->append(phase.at(pha)+":"+list[8]);
    }
//    qDebug()<<"读取表观辐亮度完毕";


    //输入图像路径
    QString path = ui->fileline->text();
    //获取图像名称
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();
    //读取图像
    Mat src_image = cv::imread(path.toStdString());
    ui->log->append("读取输入图像");
//    qDebug()<<"读取图像完成";

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
//    qDebug()<<"kmeans算法完成";
    ui->log->append("聚类完成");

    Mat output = Mat::zeros(src_image.size(),src_image.type());
    for (int i = 0;i<src_image.rows;i++)
    {
        for (int j=0;j<src_image.cols;j++)
        {
            index = i*width+j;
            int label = labels.at<int>(index);//每一个像素属于哪个标签

            //表观反射率赋值
            float L = select_appar(label);

            //计算灰度值
            int DN = get_DN(L,Gain_,Bias_);

            output.at<Vec3b>(i,j)[0]=DN;
            output.at<Vec3b>(i,j)[1]=DN;
            output.at<Vec3b>(i,j)[2]=DN;


        }
    }

    //resize分辨率转换
    int new_width = width*airres_/spaceres_;
    int new_height = height*airres_/spaceres_;

    Size dsize = Size(new_width,new_height);
    Mat final;
    cv::resize(output,final,dsize,0,0,INTER_LINEAR);

////    namedWindow("output",WINDOW_AUTOSIZE);
////    imshow("output",output);
////    waitKey(0);
////    namedWindow("final",WINDOW_AUTOSIZE);
////    imshow("final",final);
////    waitKey(0);
    //保存
    QString save_folder = ui->save_path_line->text();
    string save_path = save_folder.toStdString();
    string finalpath = save_path+"/"+file_name.toStdString();
    imwrite(finalpath,final);
    //显示输出图像
    QImage* outimg = new QImage;
    outimg -> load(QString::fromStdString(finalpath));
    ui->outputimg->setPixmap(QPixmap::fromImage(*outimg));
    ui->log->append("推演完成，图像保存至：");
    ui->log->append(QString::fromStdString(finalpath));


}


void optdeduction::on_save_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}


void optdeduction::on_concentration_currentIndexChanged(int index)
{
//    QString concentration = ui->concentration->currentText();//气溶胶含量
    //如果“气溶胶含量”为“没有气溶胶”，则v_value设置为不可选
    //如果“气溶胶含量”为“输入550nm气溶胶光学厚度”，则v_value设置一个经验值，并为可编辑状态
    //如果“气溶胶含量”为“能见度”，则v_value设置一个经验值，并为可编辑状态
/*    if (index==0){
        ui->v_value->clear();
        ui->v_value->setEditable(false);

    }else */
    if (index==0){
        ui->v_value->clear();
        ui->v_value->addItem("1.95");
        ui->v_value->setEditable(true);
    }else if (index==1){
        ui->v_value->clear();
        ui->v_value->addItem("6");
        ui->v_value->setEditable(true);
    }
}


void optdeduction::on_add_ref_clicked()
{
    QString new_type_ref = ui->new_type_ref->currentText();
    QString new_value_ref = ui->new_value_ref->currentText();
    phase<<new_type_ref;
    ref<<new_value_ref;
}


//void optdeduction::on_type_o_aerosol_currentIndexChanged(const QString &arg1)
//{
//    if (arg1 =="无气溶胶"){
//        ui->concentration->setDisabled(true);
//        ui->v_value->setDisabled(true);
//    }else{
//        ui->concentration->setDisabled(false);
//        ui->v_value->setDisabled(false);
//    }
//}

