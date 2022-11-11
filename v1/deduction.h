#ifndef DEDUCTION_H
#define DEDUCTION_H

#include <QMainWindow>
#include <sardeduction.h>
#include <optdeduction.h>


namespace Ui {
class Deduction;
}

class Deduction : public QMainWindow
{
    Q_OBJECT

public:
    explicit Deduction(QWidget *parent = nullptr);
    ~Deduction();
signals:
    void deducexit();

private slots:
    void on_SAR_clicked();
    void dealsar();

    void on_Optical_clicked();
    void dealopt();

    void on_exit_clicked();

private:
    Ui::Deduction *ui;
    SARdeduction sardeduction;
    optdeduction Optdeduction;

};

#endif // DEDUCTION_H
