#include "mainwin.h"
#include "ui_mainwin.h"

MainWin::MainWin(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWin)
{
    ui->setupUi(this);

    deduchide();
    this->setWindowTitle("被识别概率计算软件");
    this->setStyleSheet("#MainWin{background-image: url(:/new/prefix1/bg5.jpg);}");
//    this->setStyleSheet("background-image: url(:/new/prefix1/Missile.jpeg);");

    connect(ui->bu_exit, &QPushButton::clicked, this, &QMainWindow::close);

    // 近远场推演模块
    connect(ui->bu_deduction, &QPushButton::clicked, this, &MainWin::show_deduc);
    connect(&sardeduction,SIGNAL(sarexit()),this,SLOT(dealsar()));
    connect(&Optdeduction,SIGNAL(optexit()),this,SLOT(dealopt()));
    connect(&Infdeduction,SIGNAL(infexit()),this,SLOT(dealinf()));
    connect(&Hyperspectral,SIGNAL(hypexit()),this,SLOT(dealhyp()));

    connect(ui->bu_detection, &QPushButton::clicked, this, &MainWin::show_detec);
    connect(ui->bu_probability, &QPushButton::clicked, this, &MainWin::show_proba);


    connect(ui->bu_detection, &QPushButton::clicked, this, &MainWin::reset_detec);
    connect(ui->bu_probability, &QPushButton::clicked, this, &MainWin::reset_proba);

}

MainWin::~MainWin()
{
    delete ui;
}

void MainWin::reset_detec()
{
    detec.init_ui();
    detec.show();
}

void MainWin::reset_proba()
{
    proba.init_ui();
    proba.show();
}


void MainWin::deducshow(){
    ui->sar_deduc->show();
    ui->opt_deduc->show();
    ui->inf_deduc->show();
    ui->hyp_deduc->show();
}
void MainWin::deduchide(){
    ui->sar_deduc->hide();
    ui->opt_deduc->hide();
    ui->inf_deduc->hide();
    ui->hyp_deduc->hide();
}

void MainWin::show_deduc()
{
    cnt=cnt+1;
    if (cnt%2==0){
        deduchide();
    }else{
        deducshow();
    }


}

void MainWin::show_detec()
{
    detec.showMaximized();
    detec.setWindowTitle("发射装备检测识别模块");
//    this->hide();
}

void MainWin::show_proba()
{
    proba.showMaximized();
    proba.setWindowTitle("多频谱目标识别概率计算");
}



void MainWin::on_sar_deduc_clicked()
{
//    sardeduction.showFullScreen();
//    sardeduction.show();
    sardeduction.showMaximized();
    sardeduction.setWindowTitle("SAR图像推演模块");
//    this->hide();
}


void MainWin::on_opt_deduc_clicked()
{
//    Optdeduction.showFullScreen();
    Optdeduction.showMaximized();
    Optdeduction.setWindowTitle("可见光图像推演模块");
//    this->hide();
}
void MainWin::on_inf_deduc_clicked()
{
    Infdeduction.showMaximized();
    Infdeduction.setWindowTitle("红外图像推演模块");
//    this->hide();
}


void MainWin::on_hyp_deduc_clicked()
{
    Hyperspectral.showMaximized();
    Hyperspectral.setWindowTitle("高光谱图像推演模块");
//    this->hide();
}


void MainWin::dealsar()
{
    sardeduction.hide();
//    show();
    deduchide();
    cnt=0;
}
void MainWin::dealopt()
{
    Optdeduction.hide();
//    show();
    deduchide();
    cnt=0;
}
void MainWin::dealinf()
{
    Infdeduction.hide();
//    show();
    deduchide();
    cnt=0;
}
void MainWin::dealhyp()
{
    Hyperspectral.hide();
//    show();
    deduchide();
    cnt=0;
}



