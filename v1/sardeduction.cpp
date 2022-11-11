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
#include <vector>

//#include <string>
//using namespace cv;
//using std::string;
using namespace std;
SARdeduction::SARdeduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SARdeduction)
{
    ui->setupUi(this);
    srand((int)time(0));




}

SARdeduction::~SARdeduction()
{
    delete ui;
}

void SARdeduction::on_sarexit_clicked()
{
    initialize();
    emit sarexit();
}



void SARdeduction::on_browse_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,"open","../","all(*.*)");
    ui->fileline->setText(path);
    //显示图像
    QImage* srcimg = new QImage;
    srcimg -> load(path);
    ui->src_image->setPixmap(QPixmap::fromImage(*srcimg).scaled(300,300));
}



//产生瑞利随机数
double generateGaussianNoise(double mean, double sigma)
{
    double v1,S,u1;
//    srand((unsigned)time(NULL));
//    srand((int)time(0));
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

//快速傅里叶变换
void fft2Image(cv::Mat& I,cv::Mat& F)
{
    int rows = I.rows;
    int cols = I.cols;
    // 满足快速傅里叶变换的最优行数和列数
    int rPadded = cv::getOptimalDFTSize(rows);
    int cPadded = cv::getOptimalDFTSize(cols);
    //左侧和下侧补零
    cv::Mat f;
    cv::copyMakeBorder(I,f,0,rPadded-rows,0,cPadded-cols,cv::BORDER_CONSTANT,cv::Scalar::all(0));
    //快速傅里叶变换（双通道，用于储存实部和虚部
    cv::dft(f,F,cv::DFT_COMPLEX_OUTPUT);
}
//计算幅度谱
void amplitudeSpectrum(cv::Mat& _srcFFT,cv::Mat& _dstSpectrum){
    //判断傅里叶变换有两个通道
    CV_Assert(_srcFFT.channels()==2);
    //分离通道
    vector<cv::Mat> FFT2Channel;
    cv::split(_srcFFT,FFT2Channel);
    //计算傅里叶变换的幅度谱
    cv::magnitude(FFT2Channel[0],FFT2Channel[1],_dstSpectrum);
}
cv::Mat graySpectrum(cv::Mat spectrum){
    cv::Mat dst;
    log(spectrum+1,dst);
    //归一化
    cv::normalize(dst,dst,0,1,cv::NORM_MINMAX);
    //为了进行灰度级显示，做类型转换
    dst.convertTo(dst,CV_8UC1,255,0);
    return dst;
}
//构建滤波器
cv::Mat createLPFilter(cv::Size size,cv::Point center, double radius)
{
    cv::Mat lpFilter = cv::Mat::zeros(size,CV_32FC1);
    int rows = size.height;
    int cols = size.width;
    //构建理想低通滤波器
    for (int r = 0;r < rows;r++)
    {
        for(int c = 0;c < cols; c++)
        {
            float norm2 = pow(abs(float(r-center.y)),2)+pow(abs(float(c-center.x)),2);
            if (sqrt(norm2)<radius)
            {
                lpFilter.at<float>(r,c)=1;
            }else{
                lpFilter.at<float>(r,c)=0;
            }
        }
    }
    return lpFilter;
}
void SARdeduction::on_run_clicked()
{
//    QString Azimuth = ui->Azimuth->text();
    float Azimuth = ui->Azimuth->text().toFloat();
    float Range = ui->Range->text().toFloat();
    float spe_Azimuth = ui->spe_azi->text().toFloat();
    float spe_Range = ui->spe_ran->text().toFloat();
    double B=ui->B->text().toDouble()*1000000;//000000;//200MHz
    double Bd = ui->Bd->text().toDouble();
    //获取图像名称
    QString path = ui->fileline->text();
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();
    QString file_suffix = fileinfo.suffix();
    QString save_folder = ui->save_path_line->text();
    QString save_name = ui->save_name_line->text();
    ui->log->clear();
    ui->log->append("参数读取完毕");

    //读图像

    cv::Mat image = cv::imread(path.toStdString());
    cv::Mat image_gray;
    cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);
//    cv::imshow("image_gray",image_gray);
//    float b = (spe_azimuth*spe_range)/(azimuth*range);
    float BA = Azimuth/spe_Azimuth;
    float BR = Range/spe_Range;
    //B:原机载SAR发射线性调频信号的带宽
    //Bd:多普勒信号带宽
    //低通滤波器带宽（截止频率）  HA：方位向，HR：距离向

    double HA = Bd*BA;
    double HR = B*BR;

    //机载图像二维傅里叶变换
    cv::Mat F_gray;

    image_gray.convertTo(F_gray,CV_32FC1,1.0,0.0);//转换数据类型
    for(int r=0;r<F_gray.rows;r++){
        for(int c = 0;c<F_gray.cols;c++){
            if((r+c)%2)
            {
                F_gray.at<float>(r,c)*=-1;
            }
        }
    }
    cv::Mat F_image;//图像的傅里叶变换
    fft2Image(F_gray,F_image);
    //幅度谱
    cv::Mat amplSpec;
    amplitudeSpectrum(F_image,amplSpec);
    cv::Mat spectrum = graySpectrum(amplSpec);
//    cv::imshow("spectrum",spectrum);

    //找到幅度谱的最大值坐标
    cv::Point maxLoc_r;
    cv::minMaxLoc(spectrum,NULL,NULL,NULL,&maxLoc_r);

    //构建距离向低通滤波器
    cv::Mat lpFilter_R = createLPFilter(F_image.size(),maxLoc_r,HR);
    //低通滤波器和图像的傅里叶变化点乘
    cv::Mat F_lpFilter_R;
    F_lpFilter_R.create(F_image.size(),F_image.type());
    for (int r =0;r<F_lpFilter_R.rows ;r++ ) {
        for (int c = 0;c<F_lpFilter_R.cols;c++){
            //分别取出当前位置的快速傅里叶变换和理想低通滤波器的值
            cv::Vec2f F_rc = F_image.at<cv::Vec2f>(r,c);
            float lpFilter_rc = lpFilter_R.at<float>(r,c);
            //低通滤波器和图像的快速傅里叶变换的对应位置相乘；
            F_lpFilter_R.at<cv::Vec2f>(r,c)=F_rc*lpFilter_rc;
        }

    }
    //低通傅里叶变换的傅里叶谱
    cv::Mat FlpSpectrum_R;
    amplitudeSpectrum(F_lpFilter_R,FlpSpectrum_R);
    //低通傅里叶谱的灰度级显示
    FlpSpectrum_R = graySpectrum(FlpSpectrum_R);

    //转置
    cv::Mat FlpSpectrum_R_T;
    cv::transpose(FlpSpectrum_R,FlpSpectrum_R_T);
    //
    //找到幅度谱的最大值坐标
    cv::Point maxLoc_a;
    cv::minMaxLoc(FlpSpectrum_R_T,NULL,NULL,NULL,&maxLoc_a);
    //构建方位向低通滤波器
    cv::Mat lpFilter_A = createLPFilter(F_lpFilter_R.size(),maxLoc_a,HA);
    //低通滤波器和图像的傅里叶变化点乘
    cv::Mat F_lpFilter_A;
    F_lpFilter_A.create(F_lpFilter_R.size(),F_lpFilter_R.type());
    for (int r =0;r<F_lpFilter_A.rows ;r++ ) {
        for (int c = 0;c<F_lpFilter_A.cols;c++){
            //分别取出当前位置的快速傅里叶变换和理想低通滤波器的值
            cv::Vec2f F_rc_A = F_lpFilter_R.at<cv::Vec2f>(r,c);
            float lpFilter_rc_A = lpFilter_A.at<float>(r,c);
            //低通滤波器和图像的快速傅里叶变换的对应位置相乘；
            F_lpFilter_A.at<cv::Vec2f>(r,c)=F_rc_A*lpFilter_rc_A;
        }

    }

    //对低通傅里叶变换执行傅里叶逆变换，并只取实部
    cv::Mat result;
    cv::dft(F_lpFilter_A,result,CV_HAL_DFT_SCALE+CV_HAL_DFT_INVERSE+CV_HAL_DFT_REAL_OUTPUT);
    //乘(-1)^(r+c)
    for (int r =0;r<result.rows ;r++ ) {
        for (int c = 0;c<result.cols;c++){
            if((r+c)%2)
            {
                result.at<float>(r,c)*=-1;
            }
        }

    }
    result.convertTo(result,CV_8UC1,1.0,0);
    //截取左上部分，其大小与输入图像大小相同
    result=result(cv::Rect(0,0,image_gray.cols,image_gray.rows)).clone();
    ui->log->append("空间分辨率转换完成");
    //添加噪声
    cv::Mat image_output;
    Add_RayleighNoise(result,image_output,0,30,1);
    ui->log->append("噪声特性处理完成");
//    Add_RayleighNoise(image_tran,image_output,0,10,1);
    //保存
    std::string finalpath;
    std::string save_path = save_folder.toStdString();
    qDebug()<<save_name;
    if (save_name==""){
        finalpath = save_path+"/"+file_name.toStdString();
        qDebug()<<QString::fromStdString(finalpath);
    }else{
        finalpath = save_path+"/"+save_name.toStdString()+"."+file_suffix.toStdString();
        qDebug()<<QString::fromStdString(finalpath);
    }
    cv::imwrite(finalpath,image_output);
//    cv::imwrite(finalpath,image_tran);
    //显示输出图像
    QImage* outimg = new QImage;
    outimg -> load(QString::fromStdString(finalpath));
    ui->output_img->setPixmap(QPixmap::fromImage(*outimg).scaled(300,300));
    ui->log->append("SAR图像推演完成，图像保存至：");
    ui->log->append(QString::fromStdString(finalpath));

}

void SARdeduction::on_save_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}

void SARdeduction::initialize(){
    ui->B->setText("200");
    ui->Bd->setText("1389");
    ui->Azimuth->setText("1.8");
    ui->Range->setText("2");
    ui->spe_azi->setText("2");
    ui->spe_ran->setText("2");
    ui->save_path_line->clear();
    ui->save_name_line->clear();
    ui->fileline->clear();
    ui->log->clear();
    ui->src_image->clear();
    ui->output_img->clear();
}
void SARdeduction::on_bu_initialize_clicked()
{
    initialize();
}

