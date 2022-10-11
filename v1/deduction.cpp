#include "deduction.h"
#include "ui_deduction.h"

Deduction::Deduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Deduction)
{
    ui->setupUi(this);
    connect(&sardeduction,SIGNAL(sarexit()),this,SLOT(dealsar()));
    connect(&Optdeduction,SIGNAL(optexit()),this,SLOT(dealopt()));
}

Deduction::~Deduction()
{
    delete ui;

}

void Deduction::on_SAR_clicked()
{
    sardeduction.show();
    this->hide();
}
void Deduction::dealsar()
{
    sardeduction.hide();
    show();
}

void Deduction::on_Optical_clicked()
{
    Optdeduction.show();
    this->hide();
}
void Deduction::dealopt()
{
    Optdeduction.hide();
    show();
}


void Deduction::on_exit_clicked()
{
    emit deducexit();
}

