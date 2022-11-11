#include "mainwin.h"
#include "ui_mainwin.h"

MainWin::MainWin(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWin)
{
    ui->setupUi(this);

    ui->frame->hide();
    this->setWindowTitle("被识别概率计算软件");

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




void MainWin::show_deduc()
{
    cnt=cnt+1;
    if (cnt%2==0){
        ui->frame->hide();
        ui->verticalSpacer_3->changeSize(20,30,QSizePolicy::Fixed,QSizePolicy::Fixed);
    }else{
        ui->frame->show();
        ui->verticalSpacer_3->changeSize(0,0,QSizePolicy::Fixed,QSizePolicy::Fixed);
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
    ui->frame->hide();
    ui->verticalSpacer_3->changeSize(20,30,QSizePolicy::Fixed,QSizePolicy::Fixed);
    cnt=0;
}
void MainWin::dealopt()
{
    Optdeduction.hide();
//    show();
    ui->frame->hide();
    ui->verticalSpacer_3->changeSize(20,30,QSizePolicy::Fixed,QSizePolicy::Fixed);
    cnt=0;
}
void MainWin::dealinf()
{
    Infdeduction.hide();
//    show();
    ui->frame->hide();
    ui->verticalSpacer_3->changeSize(20,30,QSizePolicy::Fixed,QSizePolicy::Fixed);
    cnt=0;
}
void MainWin::dealhyp()
{
    Hyperspectral.hide();
//    show();
    ui->frame->hide();
    ui->verticalSpacer_3->changeSize(20,30,QSizePolicy::Fixed,QSizePolicy::Fixed);
    cnt=0;
}



