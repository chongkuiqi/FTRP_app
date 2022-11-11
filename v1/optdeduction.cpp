#include "util.h"
#include "optdeduction.h"
#include "ui_optdeduction.h"
#include <QtDebug>
#include <iostream>

//#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <cmath>
#include <QDir>
#include <QCompleter>
#include <QStringList>
#include <QProcess>
#include <QFile>
#include <algorithm>
#include <QDateTime>

//#include <dlfcn.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string>
using namespace cv;
using namespace std;
using std::string;
float normalize_mean[3] = {0.485,0.456,0.406};
float normalize_std[3] = {0.229, 0.224, 0.225};
optdeduction::optdeduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::optdeduction)
{
    ui->setupUi(this);
//    phase<<"vegetation"<<"clear_water"<<"dry_sand"<<"lake_water"<<"volcanic_debris"<<"road"<<"land"<<"grass";
    phase<<"road"<<"forrest"<<"grass"<<"snow"<<"water"<<"sand";
//    ref<<"0.09"<<"0.15"<<"0.2";
    ui->label_road->setStyleSheet("QLabel{background-color:rgb(255,255,0);}");
    ui->label_grass->setStyleSheet("QLabel{background-color:rgb(255,195,128);}");
    ui->label_forrest->setStyleSheet("QLabel{background-color:rgb(0,255,0);}");
    ui->label_snow->setStyleSheet("QLabel{background-color:rgb(255,255,255);}");
    ui->label_sand->setStyleSheet("QLabel{background-color:rgb(159,129,183);}");
//    ui->label_water->setStyleSheet("QLabel{background-color:rgb(0,0,255);}");
    ui->label_water->setStyleSheet("QLabel{background-color:rgb(135,206,255);}");


}

optdeduction::~optdeduction()
{
    delete ui;
}

void optdeduction::on_exit_clicked()
{
    initialize();
    emit optexit();
}


void optdeduction::on_browse_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,"open","../","all(*.*)");
    ui->fileline->setText(path);
    //显示图像
    QImage* srcimg = new QImage;
    srcimg -> load(path);
//    srcimg->scaled(10,10);
    ui->srcimg->setPixmap(QPixmap::fromImage(*srcimg).scaled(250,250));
}


int optdeduction::select_DN(int label)
{
    int id;
    switch(label)
    {
        case 0:
            id = phase.indexOf("sand");//背景->荒地
//            return appar.at(id).toFloat();c
            return DN_norm[id];
        case 1:
            id = phase.indexOf("sand");//建筑->荒地
//            return appar.at(id).toFloat();
            return DN_norm[id];
        case 2:
            id = phase.indexOf("road");//公路
//            return appar.at(id).toFloat();
            return DN_norm[id];
        case 3:
            id = phase.indexOf("water");//水
//            return appar.at(id).toFloat();
            return DN_norm[id];
        case 4:
            id = phase.indexOf("sand");//荒地
//            return appar.at(id).toFloat();
            return DN_norm[id];
        case 5:
            id = phase.indexOf("forrest");//森林
//            return appar.at(id).toFloat();
            return DN_norm[id];

        case 6:
            id = phase.indexOf("grass");//草地
//            return appar.at(id).toFloat();
            return DN_norm[id];


    }

}

void optdeduction::label2rgb(int label)
{
    bgr.clear();
    switch(label)
    {
        case 0:
//            rgb <<"255"<<"255"<<"255";
            bgr <<"159"<<"129"<<"183";//背景->荒地
        case 1:
//            rgb <<"255"<<"0"<<"0";//
            bgr <<"159"<<"129"<<"183";//建筑->荒地
        case 2:
            bgr <<"255"<<"255"<<"0";//公路
        case 3:
//            bgr <<"0"<<"0"<<"255";//水
            bgr <<"130"<<"206"<<"255";//水
        case 4:
            bgr <<"159"<<"129"<<"183";//荒地
        case 5:
            bgr <<"0"<<"255"<<"0";//森林
        case 6:
            bgr <<"255"<<"195"<<"128";//草地

    }

}

void optdeduction::get_DN(float Gain,float Bias)
{
//    int Bias = 1;
//    int DN = (L-Bias)/Gain;
    for (int i=0;i<6 ;++i ) {
        DN[i] = (appar.at(i).toFloat()-Bias)/Gain;
    }
    qDebug()<<DN[0]<<" "<<DN[1]<<" "<<DN[2]<<" "<<DN[3]<<" "<<DN[4]<<" "<<DN[5];
    int DN_max;
    int DN_min;
    DN_max= *max_element(DN,DN+6);
    DN_min= *min_element(DN,DN+6);
    qDebug()<<DN_max;
//    //归一到0～255
    int d_max = DN_max-230;
    int d_min = DN_min-10;
    if (d_max>0){
        for (int i=0;i<6 ;++i ) {
            DN_norm[i] = DN[i]-d_max;
            if (DN_norm[i]<0){
                DN_norm[i] = DN[i]-d_min;
            }
        }
    }else{
        for (int i=0;i<6 ;++i ) {
            DN_norm[i] = DN[i];

        }
    }




}


void optdeduction::gen_input_txt(QString solar_zenith_angle,QString solar_azimuth,QString satellite_zenith_angle,QString satellite_aximuth\
                   ,int atmospheric_model,int type_o_aerosol\
                   ,QString altitude_of_target,int spectral_conditions){
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
        out<<solar_zenith_angle<<" "<<solar_azimuth<<" "<<satellite_zenith_angle<<" "<<satellite_aximuth<<" "<<"1"<<" "<<"1"<<"\n";
        //2 大气模式
        out<<QString::number(atmospheric_model+1,10)<<"\n";
        //3 气溶胶类型
        out<<QString::number(type_o_aerosol+1,10)<<"\n";
        //4 气溶胶含量
        out<<"0\n"<<"1.95"<<"\n";
        //5 目标高度
        if (altitude_of_target=="0"){
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
        out<<"0\n"<<ref.at(i)<<"\n";

        //不激活大气校正方式
        out<<"-2\n";
        input_txt.close();
    }
    sh.close();
}



void optdeduction::on_classify_clicked()
{
//    ui->srcimg->clear();
    ui->tempimg->clear();
    ui->outputimg->clear();
    //数组初始化
    appar.clear();
    auto seed = QDateTime::currentDateTime().toMSecsSinceEpoch();
    qsrand(seed);


    //参数读取
    //观测辐亮度计算参数
    QString solar_zenith_angle = ui->solar_zenith_angle->text();//太阳天顶角
    QString solar_azimuth = ui->solar_azimuth->text();//太阳方位角
    QString satellite_zenith_angle = ui->satellite_zenith_angle->text();//卫星天顶角
    QString satellite_aximuth = ui->satellite_aximuth->text();//卫星方位角
    int atmospheric_model = ui->atmospheric_model->currentIndex();//大气模式
    int type_o_aerosol = ui->type_o_aerosol->currentIndex();//气溶胶模式
    QString altitude_target = ui->altitude_target->text();//目标高度
    int spectral_conditions = ui->spectral_conditions->currentIndex();//光谱参数
    //反射率
    QString road = ui->road->text();
    QString forrest = ui->forrest->text();
    QString grass = ui->grass->text();
    QString snow = ui->snow->text();
    QString water = ui->water->text();
    QString sand = ui->sand->text();
    ref<<road<<forrest<<grass<<snow<<water<<sand;

    //辐射定标计算参数
    QString Gain = ui->gain->text();
    QString Bias = ui->bias->text();
    float Gain_ = Gain.toFloat();
    float Bias_ = Bias.toFloat();
    //分辨率转换参数
    QString airres = ui->airres->text();
    QString spaceres = ui->spaceres->text();
    float airres_ = airres.toFloat();
    float spaceres_ = spaceres.toFloat();
    ui->log->clear();
    ui->log->append("参数读取完毕");

    QApplication::processEvents();
    qDebug()<<"参数读取完毕";

    //生成input.txt
    gen_input_txt(solar_zenith_angle,solar_azimuth,satellite_zenith_angle,satellite_aximuth\
                  ,atmospheric_model,type_o_aerosol\
                  ,altitude_target,spectral_conditions);

    //调用6S.exe
//    qDebug()<<"调用6S";
    QProcess myProcess;
    myProcess.execute("./6s.sh");
//    qDebug()<<myProcess.error();
//    qDebug()<<"表观辐亮度计算完毕";
    ui->log->append("表观辐亮度计算完毕");
    QApplication::processEvents();
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
        qDebug()<<phase.at(pha)+":"+list[8];

    }
    get_DN(Gain_,Bias_);

    //输入图像路径
    QString path = ui->fileline->text();
    //获取图像名称
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();
    QString file_suffix = fileinfo.suffix();

    //读取图像
    Mat src_image = cv::imread(path.toStdString());
    ui->log->append("读取输入图像");
    QApplication::processEvents();
    int width = src_image.cols;
    int height = src_image.rows;


//    语义分割
    torch::Device device(torch::kCPU);
    std::cout << "cuda is_available:" << torch::cuda::is_available() << std::endl;
    ui->log->append("正在获取图像和模型...");
    QApplication::processEvents();
    at::Tensor input_tensor;
    cv::cvtColor(src_image, src_image, cv::COLOR_BGR2RGB);

    src_image.convertTo(src_image, CV_32FC3);

    input_tensor = torch::from_blob(
      src_image.data, {1, IMG_SIZE, IMG_SIZE, IMG_CHN}).toType(torch::kFloat32);
    input_tensor = input_tensor.permute({0, 3, 1, 2});

    input_tensor[0][0] = input_tensor[0][0].div(255).sub_(normalize_mean[0]).div_(normalize_std[0]);
    input_tensor[0][1] = input_tensor[0][1].div(255).sub_(normalize_mean[1]).div_(normalize_std[1]);
    input_tensor[0][2] = input_tensor[0][2].div(255).sub_(normalize_mean[2]).div_(normalize_std[2]);
    input_tensor = input_tensor.to(device);

    std::string model_path = "./best_script.pt";
    torch::NoGradGuard no_grad;
    torch::jit::script::Module model;
    model = torch::jit::load(model_path, device);

    at::Tensor output = model.forward({input_tensor}).toTensor();

    output = torch::squeeze(output);
    output = torch::argmax(output,0);
    ui->log->append("语义分割完成");
    QApplication::processEvents();

    //将聚类后的结果显示出来
    Mat output_seg = Mat::zeros(src_image.size(),CV_8UC3);
//    qDebug()<<src_image.size;
    Mat output_deduc = Mat::zeros(Size(width,height),CV_8UC1);


    for (int i = 0;i<src_image.rows;i++)
        {
            for (int j=0;j<src_image.cols;j++)

            {
                int label = output[i][j].item<int>();//每一个像素属于哪个标签
                //聚类图像
                label2rgb(label);
                output_seg.at<Vec3b>(i,j)[0]=bgr.at(2).toInt();
                output_seg.at<Vec3b>(i,j)[1]=bgr.at(1).toInt();
                output_seg.at<Vec3b>(i,j)[2]=bgr.at(0).toInt();
                //表观反射率赋值
                int current_DN = select_DN(label);
                //计算灰度值
//                int DN = get_DN(L,Gain_,Bias_);
//                get_DN(Gain_,Bias_);
                //星载图像

                output_deduc.at<uchar>(i,j)=current_DN+qrand()%20;
//                output_deduc.at<unsigned short>(i,j)=(unsigned short)current_DN;

            }
        }

        QString save_folder = ui->save_path_line->text();
        string save_path = save_folder.toStdString();
        string seg_finalpath = save_path+"/"+"seg_"+file_name.toStdString();
        imwrite(seg_finalpath,output_seg);
        QImage* temp = new QImage;
        temp -> load(QString::fromStdString(seg_finalpath));
        ui->tempimg->setPixmap(QPixmap::fromImage(*temp).scaled(250,250));

        //resize分辨率转换
        int new_width = width*airres_/spaceres_;
        int new_height = height*airres_/spaceres_;


        Size dsize = Size(new_width,new_height);
        Mat final;
        cv::resize(output_deduc,final,dsize,0,10,INTER_LINEAR);
        ui->log->append("分辨率转换完成");
        QApplication::processEvents();

        QString save_name = ui->save_name->text();
        std::string finalpath;
        if (save_name.toStdString()==""){
            finalpath = save_path+"/"+file_name.toStdString();
        }else{
            finalpath = save_path+"/"+save_name.toStdString()+"."+file_suffix.toStdString();
        }
        imwrite(finalpath,final);
//        imwrite(finalpath,output_deduc);
        ui->log->append("可见光图像推演完成，图像保存至：");
        ui->log->append(QString::fromStdString(finalpath));
        QApplication::processEvents();
        QImage* out = new QImage;
        out -> load(QString::fromStdString(finalpath));
        ui->outputimg->setPixmap(QPixmap::fromImage(*out).scaled(250,250));


}


void optdeduction::on_save_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}

void optdeduction::initialize(){
    ui->fileline->clear();
    ui->solar_zenith_angle->setText("60");
    ui->solar_azimuth->setText("180");
    ui->satellite_zenith_angle->setText("60");
    ui->satellite_aximuth->setText("0");
    ui->atmospheric_model->setCurrentIndex(0);
    ui->type_o_aerosol->setCurrentIndex(0);
    ui->altitude_target->setText("0");
    ui->spectral_conditions->setCurrentIndex(0);
    ui->gain->setText("0.1887");
    ui->bias->setText("0");
    ui->road->setText("0.09");
    ui->forrest->setText("0.32");
    ui->grass->setText("0.25");
    ui->snow->setText("0.83");
    ui->water->setText("0.05");
    ui->sand->setText("0.21");
    ui->save_path_line->clear();
    ui->save_name->clear();
    ui->srcimg->clear();
    ui->tempimg->clear();
    ui->outputimg->clear();
    ui->log->clear();

}







void optdeduction::on_initialize_clicked()
{
    initialize();
}

