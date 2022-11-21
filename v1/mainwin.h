#ifndef MAINWIN_H
#define MAINWIN_H

#include <QMainWindow>
#include<deduction.h>
#include <sardeduction.h>
#include <optdeduction.h>
#include <infdeduction.h>
#include <hyperspectral.h>

#include<detection.h>
#include<probability.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWin; }
QT_END_NAMESPACE

class MainWin : public QMainWindow
{
    Q_OBJECT

public:
    MainWin(QWidget *parent = nullptr);
    ~MainWin();

    void show_deduc();
    void show_detec();
    void show_proba();

    void reset_detec();
    void reset_proba();
    void deducshow();
    void deduchide();

private slots:
    void on_sar_deduc_clicked();

    void on_opt_deduc_clicked();
    void dealsar();
    void dealopt();
    void dealinf();
    void dealhyp();

    void on_inf_deduc_clicked();

    void on_hyp_deduc_clicked();

private:
    Ui::MainWin *ui;


    Detection detec;
    Probability proba;

    // Deduction deduc;
    SARdeduction sardeduction;
    optdeduction Optdeduction;
    infdeduction Infdeduction;
    hyperspectral Hyperspectral;
    int cnt=0;

};
#endif // MAINWIN_H
