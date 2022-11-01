#include "mainwin.h"
#include "ui_mainwin.h"

MainWin::MainWin(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWin)
{
    ui->setupUi(this);

    connect(ui->bu_exit, &QPushButton::clicked, this, &QMainWindow::close);

    // 近远场推演模块
    connect(ui->bu_deduction, &QPushButton::clicked, this, &MainWin::show_deduc);
    connect(&deduc,SIGNAL(deducexit()),this,SLOT(dealdeduc()));

    connect(ui->bu_detection, &QPushButton::clicked, this, &MainWin::show_detec);
    connect(ui->bu_probability, &QPushButton::clicked, this, &MainWin::show_proba);

}

MainWin::~MainWin()
{
    delete ui;
}

void MainWin::show_deduc()
{
    deduc.show();
}

void MainWin::show_detec()
{
    detec.init_ui();
    detec.show();
}

void MainWin::show_proba()
{

    proba.show();
}

void MainWin::dealdeduc()
{
    deduc.hide();
    show();
}
