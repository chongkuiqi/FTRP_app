#ifndef DETECTION_H
#define DETECTION_H

#include <QMainWindow>

namespace Ui {
class Detection;
}

class Detection : public QMainWindow
{
    Q_OBJECT

public:
    explicit Detection(QWidget *parent = nullptr);
    ~Detection();

    void browse_img();
    void browse_model();

    void detect();

private:
    Ui::Detection *ui;
};

#endif // DETECTION_H
