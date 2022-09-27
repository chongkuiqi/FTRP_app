#include "deduction.h"
#include "ui_deduction.h"

Deduction::Deduction(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Deduction)
{
    ui->setupUi(this);
}

Deduction::~Deduction()
{
    delete ui;
}
