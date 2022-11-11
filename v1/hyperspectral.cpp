#include "hyperspectral.h"
#include "ui_hyperspectral.h"

#include <QFileDialog>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
using namespace std;
float transmitance[12]={0.7,0.95,0.8,0.6,0.95,0.9,0.8,0.65,0.8,0.4,0.6,1};//大气窗口
//0.3~0.4 0.7
//0.4~0.7 0.95
//0.7~1.1 0.8
//1.4~1.55 0.6
//1.55~1.75 0.95
//1.75~1.9 0.9
//1.9~2,5 0.8
//3.5~5 0.65
//8~14 0.8
//1~1.8mm 0.4
//2~5mm 0.6
//8~1000mm 1
hyperspectral::hyperspectral(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::hyperspectral)
{
    ui->setupUi(this);
}

hyperspectral::~hyperspectral()
{
    delete ui;
}
//选择对应波长的大气透过率
float select_transmitance(float spec){
    if(spec<0.4 && spec>=0.3){
        return transmitance[0];
    }else if(spec<0.7){
        return transmitance[1];
    }else if(spec<1.1){
        return transmitance[2];
    }else if(spec<1.55&&spec>1.4){
        return transmitance[3];
    }else if(spec<1.75){
        return transmitance[4];
    }else if(spec<1.9){
        return transmitance[5];
    }else if(spec<2.5){
        return transmitance[6];
    }else if(spec<5&&spec>3.5){
        return transmitance[7];
    }else if(spec<14&&spec>8){
        return transmitance[8];
    }else{
        return 0;
    }
}

void hyperspectral::on_browser_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,"open","../","all(*.*)");
    ui->fileline->setText(path);
}

//读取csv
vector<vector<double>> hyperspectral::read_csv_file(string filename){
    ifstream inFile(filename,ios::in);
    string lineStr;
    char delim = ',';
    vector<vector<double>> matrix;
    if (inFile.fail()){
        cout<<"Read files fail ......"<<endl;
        return matrix;
    }else{
        while (getline(inFile,lineStr)){
            stringstream ss(lineStr);
            string str;
            vector<double> lineArray;
            int i=0;
            while(getline(ss,str,delim)){
                i++;
                lineArray.push_back(std::stod(str));
            }
            matrix.push_back(lineArray);
        }
        return matrix;

    }
    cout<<matrix[0].size()<<endl;
}


void hyperspectral::on_preview_clicked()
{
   out_image_vec.clear();
   string path = ui->fileline->text().toStdString();
   //显示图像
   //读取csv文件
   hyper_matrix = read_csv_file(path);
   ui->log->clear();

   //2Dto3D
//   vector<cv::Mat> image_vec;
   image_vec.clear();
   int height=ui->height->text().toInt();
   int width =ui->width->text().toInt();
   int num_spec = ui->num_spec->text().toInt();
//   cout<<hyper_matrix[0][0]<<endl;
   for (int ch=0; ch<num_spec;ch++ ) {
       cv::Mat image=cv::Mat(cv::Size(width,height),CV_32FC1);
       for (int r=0;r<height;r++){
           for(int c=0;c<width;c++){
               image.at<float>(r,c)=hyper_matrix[r][c+ch*width];
           }
       }
       image_vec.push_back(image);
   }
   //显示第一副图像
   QImage img;
   cv::Mat normalize_mat;
   cv::Mat first_image;
   first_image = image_vec[0];
   cv::normalize(first_image,normalize_mat,0,255,cv::NORM_MINMAX,-1);
   normalize_mat.convertTo(normalize_mat,CV_8U);
   const uchar *pSrc = (const uchar*)normalize_mat.data;
   img=QImage(pSrc,normalize_mat.cols,normalize_mat.rows,normalize_mat.step,QImage::Format_Grayscale8);
   ui->srcimg->setPixmap(QPixmap::fromImage(img).scaled(300,300));
//   cout<<first_image.at<float>(0,0)<<endl;

//   cout<<image_vec[0].at<float>(0,0)<<endl;
//   cout<<image_vec[2].at<float>(0,0)<<endl;
   ui->srcSlider->setEnabled(true);
   ui->total->setText(ui->num_spec->text());
   ui->current->setText("1");
   ui->srcSlider->setRange(1,num_spec);
   ui->srcSlider->setValue(1);
   ui->log->append("读取机载高光谱图像数据完毕");
   ui->log->append("请拖动滑钮进行预览");
   ui->run->setEnabled(true);

}


void hyperspectral::on_srcSlider_sliderMoved(int position)
{

    if (out_image_vec.size()==0){
        cv::Mat current_image;
        current_image = image_vec[position-1];
        QImage img;
        cv::Mat normalize_mat;
        cv::normalize(current_image,normalize_mat,0,255,cv::NORM_MINMAX,-1);
        normalize_mat.convertTo(normalize_mat,CV_8U);
        const uchar *pSrc = (const uchar*)normalize_mat.data;
        img=QImage(pSrc,normalize_mat.cols,normalize_mat.rows,normalize_mat.step,QImage::Format_Grayscale8);
        ui->srcimg->setPixmap(QPixmap::fromImage(img).scaled(300,300));
    //    cout<<position<<endl;
        ui->current->setText(QString::number(position,10));
    }else{
        cv::Mat current_image;
        current_image = image_vec[position-1];
        QImage img;
        cv::Mat normalize_mat;
        cv::normalize(current_image,normalize_mat,0,255,cv::NORM_MINMAX,-1);
        normalize_mat.convertTo(normalize_mat,CV_8U);
        const uchar *pSrc = (const uchar*)normalize_mat.data;
        img=QImage(pSrc,normalize_mat.cols,normalize_mat.rows,normalize_mat.step,QImage::Format_Grayscale8);
        ui->srcimg->setPixmap(QPixmap::fromImage(img).scaled(300,300));
    //    cout<<position<<endl;
        ui->current->setText(QString::number(position,10));
        float min = ui->min->text().toFloat();
        float max = ui->max->text().toFloat();
        int num_spec = ui->num_spec->text().toFloat();
        float step = (max-min)/num_spec;
        float current_spec = (position-1)*step+min;
        cv::Mat current_out_image;
        current_out_image = out_image_vec[position-1];
        QImage out_img;
        cv::Mat normalize_out_mat;
        cv::normalize(current_out_image,normalize_out_mat,0,(int)255*select_transmitance(current_spec),cv::NORM_MINMAX,-1);
        normalize_out_mat.convertTo(normalize_out_mat,CV_8U);
        const uchar *pSrc_out = (const uchar*)normalize_out_mat.data;
        out_img=QImage(pSrc_out,normalize_out_mat.cols,normalize_out_mat.rows,normalize_out_mat.step,QImage::Format_Grayscale8);
        ui->outimg->setPixmap(QPixmap::fromImage(out_img).scaled(300,300));

    }



}


void hyperspectral::on_run_clicked()
{
    out_image_vec.clear();
    float min = ui->min->text().toFloat();
    float max = ui->max->text().toFloat();
    int num_spec = ui->num_spec->text().toFloat();
    float step = (max-min)/num_spec;
    float current_spec = min;
    float res = ui->res->text().toFloat();
    float space_res = ui->space_res->text().toFloat();
    int width = ui->width->text().toInt();
    int height = ui->height->text().toInt();
    int new_width = width*res/space_res;
    int new_height = height*res/space_res;

    QFileInfo fileinfo = QFileInfo(ui->fileline->text());
    QString file_name = fileinfo.fileName();
    QString file_suffix = fileinfo.suffix();
    string save_dir = ui->save_path_line->text().toStdString();
    string save_name = ui->save_name->text().toStdString();

    ui->log->append("参数读取完毕");
    QApplication::processEvents();

    for(int i=0;i<(int)num_spec;i++){
        float current_trans = select_transmitance(current_spec);
        cv::Mat out_image;
        out_image = image_vec[i]*current_trans;
        cv::Mat final;
//        cv::resize(out_image,final,cv::Size(new_width,new_height),0,0,cv::INTER_LINEAR);
        cv::resize(out_image,final,cv::Size(new_width,new_height),0,0,cv::INTER_NEAREST);
        out_image_vec.push_back(final);
        current_spec+=step;

    }
    ui->log->append("幅值特征转换完毕");
    ui->log->append("空间分辨率转换完毕");
    QApplication::processEvents();
    cout<<out_image_vec[0].at<float>(0,0)<<endl;
    int position = ui->srcSlider->value();
    current_spec = (position-1)*step+min;
    QImage img;
    cv::Mat normalize_mat;
    cv::Mat first_image;
    first_image = out_image_vec[position-1];
    cv::normalize(first_image,normalize_mat,0,(int)255*select_transmitance(current_spec),cv::NORM_MINMAX,-1);
    normalize_mat.convertTo(normalize_mat,CV_8U);
    const uchar *pSrc = (const uchar*)normalize_mat.data;
    img=QImage(pSrc,normalize_mat.cols,normalize_mat.rows,normalize_mat.step,QImage::Format_Grayscale8);
    ui->outimg->setPixmap(QPixmap::fromImage(img).scaled(300,300));
    ui->log->append("高光谱图像推演完成，请拖动滑钮进行预览");
    QApplication::processEvents();
    //保存
    string final_path;
    if (save_name==""){
        final_path=save_dir+"/"+file_name.toStdString();

    }else{
        final_path=save_dir+"/"+save_name+"."+file_suffix.toStdString();
    }
    ofstream outFile(final_path,ios::out);
    if(!outFile){
        cout<<"打开文件失败"<<endl;
    }
    cout<<new_height<<new_width<<endl;
    for (int r =0 ;r<new_height ;r++ ) {
        for(int ch = 0;ch<num_spec;ch++){
            for(int c = 0;c<new_width;c++){
                outFile<<out_image_vec[ch].at<float>(r,c)<<",";
            }
        }
        outFile<<endl;
    }
    cout<<"写入完成"<<endl;
    ui->log->append("图像数据保存至：");
    ui->log->append(QString::fromStdString(final_path));
    QApplication::processEvents();

}





void hyperspectral::on_browser_2_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}

void hyperspectral::initialize(){
    ui->log->clear();
    ui->fileline->clear();
    ui->width->setText("340");
    ui->height->setText("610");
    ui->min->setText("0.4");
    ui->max->setText("0.86");
    ui->num_spec->setText("103");
    ui->res->setText("1.3");
    ui->space_res->setText("2");
    ui->save_path_line->clear();
    ui->save_name->clear();
    ui->srcSlider->setValue(1);
    ui->srcSlider->setEnabled(false);
    ui->srcimg->clear();
    ui->outimg->clear();
    ui->current->setText("当前波段");
    ui->total->setText("总波段数");
    out_image_vec.clear();
    image_vec.clear();
    hyper_matrix.clear();
    ui->run->setEnabled(false);
}


void hyperspectral::on_initialize_clicked()
{
    initialize();
}


void hyperspectral::on_exit_clicked()
{
    initialize();
    emit hypexit();
}


void hyperspectral::on_fileline_textChanged(const QString &arg1)
{
    ui->run->setEnabled(false);
}


void hyperspectral::on_width_textChanged(const QString &arg1)
{
    ui->run->setEnabled(false);
}


void hyperspectral::on_height_textChanged(const QString &arg1)
{
    ui->run->setEnabled(false);
}


void hyperspectral::on_num_spec_textChanged(const QString &arg1)
{
    ui->run->setEnabled(false);
}

