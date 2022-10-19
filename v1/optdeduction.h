#ifndef OPTDEDUCTION_H
#define OPTDEDUCTION_H

#include <QMainWindow>
#include <QFileDialog>

namespace Ui {
class optdeduction;
}

class optdeduction : public QMainWindow
{
    Q_OBJECT

public:
    explicit optdeduction(QWidget *parent = nullptr);
    ~optdeduction();
signals:
    void optexit();

private slots:
    void on_exit_clicked();

    void on_browse_clicked();


    void on_classify_clicked();

    void on_save_clicked();

    void on_concentration_currentIndexChanged(int index);

    void on_add_ref_clicked();


//    void on_type_o_aerosol_currentIndexChanged(const QString &arg1);

private:
    void gen_input_txt(QString solar_zenith_angle,QString solar_azimuth,QString satellite_zenith_angle,QString satellite_aximuth,QString month\
                       ,QString date,int atmospheric_model,int type_o_aerosol,int concentration,QString v_value\
                       ,QString altitude_of_target,int spectral_conditions);
    float select_appar(int label);
    QStringList phase;
    QStringList ref;
    QStringList appar;

    Ui::optdeduction *ui;

};

#endif // OPTDEDUCTION_H
