#include "infdeduction.h"
#include "ui_infdeduction.h"
#include <QtDebug>
#include <QDir>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <QStringList>
#include <QLabel>
#include <QMessageBox>
float transmittance[3] = {0.8, 0.7, 0.8};
infdeduction::infdeduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::infdeduction)
{
    ui->setupUi(this);
}

infdeduction::~infdeduction()
{
    delete ui;
}

void infdeduction::on_browser_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,"open","../","all(*.*)");
    ui->fileline->setText(path);
    //显示图像
    QImage* srcimg = new QImage;
    srcimg -> load(path);
//    srcimg->scaled(10,10);
    ui->src_img->setPixmap(QPixmap::fromImage(*srcimg).scaled(300,300));
}


void infdeduction::on_browser_2_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    ui->save_path_line->setText(path);
}



void infdeduction::on_run_clicked()
{
    ui->log->clear();
    int phase = ui->phase->currentIndex();
    float res = ui->res->text().toFloat();
    float spe_res = ui->spe_res->text().toFloat();
    QString path = ui->fileline->text();
    QString save_dir = ui->save_path_line->text();
    QString save_name = ui->save_name->text();
    ui->log->append("参数读取完毕");
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();
    QString file_suffix = fileinfo.suffix();
    cv::Mat image = cv::imread(path.toStdString());
    cv::Mat image_gray;
    cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);
    ui->log->append("读取图像完毕");
    cv::Mat image_output;
    image_output.create(image_gray.rows,image_gray.cols,image_gray.type());
    for (int i=0;i<image_gray.rows ;i++ ) {
        for (int j = 0;j<image_gray.cols;j++)
        {
//            double temp = cv::saturate_cast<uchar>(img_input.at<uchar>(x,y)-k*generateGaussianNoise(mean,sigma));
            uchar temp = image_gray.at<uchar>(i,j)*transmittance[phase];
            image_output.at<uchar>(i,j)=temp;
        }
    }
    //resize分辨率转换
    int width = image_gray.rows;
    int height = image_gray.cols;
    int new_width = width*res/spe_res;
    int new_height = height*res/spe_res;
    ui->log->append("分辨率转换完成");
    QApplication::processEvents();
    cv::Size dsize = cv::Size(new_width,new_height);
//    cv::Mat final;
    cv::resize(image_output,final,dsize,0,10,cv::INTER_LINEAR);



    //显示输出图像
    QImage img;
//    cv::normalize(first_image,normalize_mat,0,(int)255*select_transmitance(current_spec),cv::NORM_MINMAX,-1);
    image_output.convertTo(image_output,CV_8U);
    const uchar *pSrc = (const uchar*)image_output.data;
    img=QImage(pSrc,image_output.cols,image_output.rows,image_output.step,QImage::Format_Grayscale8);
//    QImage* outimg = new QImage;
//    outimg -> load(QString::fromStdString(finalpath));
    ui->out_img->setPixmap(QPixmap::fromImage(img).scaled(300,300));
    ui->log->append("红外图像推演完成");
    ui->save->setEnabled(true);
//    ui->log->append(QString::fromStdString(finalpath));




}
void infdeduction::initialize(){
    ui->fileline->clear();
    ui->res->setText("1");
    ui->spe_res->setText("2");
    ui->log->clear();
    ui->save_name->clear();
    ui->save_path_line->clear();
    ui->src_img->clear();
    ui->out_img->clear();
    ui->save->setEnabled(false);
}
void infdeduction::on_initialize_clicked()
{
    initialize();
}


void infdeduction::on_exit_clicked()
{
    initialize();
    emit infexit();
}


void infdeduction::on_save_clicked()
{
    QString path = ui->fileline->text();
    QString save_dir = ui->save_path_line->text();
    QString save_name = ui->save_name->text();
    QFileInfo fileinfo = QFileInfo(path);
    QString file_name = fileinfo.fileName();
    QString file_suffix = fileinfo.suffix();
    std::string finalpath;
    std::string save_path = save_dir.toStdString();
    qDebug()<<save_name;
    if (save_dir==""){
        QMessageBox::warning(this,tr("错误提示"),
                             tr("请选择保存路径"),
                             QMessageBox::Ok,
                             QMessageBox::Ok
                             );
        ui->log->append("请选择保存路径");
        return ;
    }
    if (save_name==""){
        finalpath = save_path+"/"+file_name.toStdString();
        qDebug()<<QString::fromStdString(finalpath);
    }else{
        finalpath = save_path+"/"+save_name.toStdString()+"."+file_suffix.toStdString();
        qDebug()<<QString::fromStdString(finalpath);
    }
    cv::imwrite(finalpath,final);
    ui->log->append("图像保存至：");
    ui->log->append(QString::fromStdString(finalpath));
}

