from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QApplication, QRadioButton, QLabel, QLineEdit, QSpinBox, QCheckBox, QTabWidget, QVBoxLayout, QTextBrowser
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot, pyqtSignal

import numpy as np
import pandas as pd

from additive_model import get_results


def connect(obj, func):
    if isinstance(obj, QSpinBox):
        obj.valueChanged.connect(func)
    else:
        obj.clicked.connect(func)
    return obj


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.__inpuut_file = QLineEdit('data/data_var1.csv')
        self.__output_file = QLineEdit('data/output_own.csv')
        self.__sample_size = QSpinBox(value=50)
        self.__x1_dim = QSpinBox(value=2)
        self.__x2_dim = QSpinBox(value=2)
        self.__x3_dim = QSpinBox(value=3)
        self.__y_dim = QSpinBox(value=4)

        self.__chebyshev = QRadioButton('Chebyshev', checked=True)
        self.__legendre = QRadioButton('Legendre')
        self.__hermit = QRadioButton('Hermit')
        self.__laguerre = QRadioButton('Laguerre')

        self.__lambda = QCheckBox('Calculate lambda from 3 systems')

        self.__x1_degree = QSpinBox(value=2)
        self.__x2_degree = QSpinBox(value=2)
        self.__x3_degree = QSpinBox(value=2)

        self.__calc_button = QPushButton('Calculate Results', self)
        self.__calc_button.clicked.connect(self.__button_press)

        self.text_output = QTextBrowser()

        self.graphic_x1 = pg.PlotWidget()
        self.graphic_x2 = pg.PlotWidget()
        self.graphic_x3 = pg.PlotWidget()

        self.__initUI__()

    def __initUI__(self):
        input_grid = QGridLayout()
        input_grid.setVerticalSpacing(5)
        input_grid.addWidget(QLabel('Input file'), 0, 0)
        input_grid.addWidget(QLabel('Output file'), 1, 0)
        input_grid.addWidget(QLabel('Sample size'), 2, 0)
        input_grid.addWidget(QLabel('X1 dim'), 3, 0)
        input_grid.addWidget(QLabel('X2 dim'), 4, 0)
        input_grid.addWidget(QLabel('X3 dim'), 5, 0)
        input_grid.addWidget(QLabel('Y dim'), 6, 0)
        input_grid.addWidget(self.__inpuut_file, 0, 1)
        input_grid.addWidget(self.__output_file, 1, 1)
        input_grid.addWidget(self.__sample_size, 2, 1)
        input_grid.addWidget(self.__x1_dim, 3, 1)
        input_grid.addWidget(self.__x2_dim, 4, 1)
        input_grid.addWidget(self.__x3_dim, 5, 1)
        input_grid.addWidget(self.__y_dim, 6, 1)

        polynomes_grid = QGridLayout()
        polynomes_grid.addWidget(self.__chebyshev, 0, 0)
        polynomes_grid.addWidget(self.__legendre, 1, 0)
        polynomes_grid.addWidget(self.__hermit, 2, 0)
        polynomes_grid.addWidget(self.__laguerre, 3, 0)
        polynomes_grid.addWidget(self.__lambda, 4, 0)

        degree_grid_layout = QGridLayout()
        degree_grid_layout.addWidget(QLabel('X1'), 0, 0)
        degree_grid_layout.addWidget(QLabel('X2'), 1, 0)
        degree_grid_layout.addWidget(QLabel('X3'), 2, 0)
        degree_grid_layout.addWidget(self.__x1_degree, 0, 1)
        degree_grid_layout.addWidget(self.__x2_degree, 1, 1)
        degree_grid_layout.addWidget(self.__x3_degree, 2, 1)

        menu_layout = QGridLayout()
        menu_layout.setHorizontalSpacing(50)
        menu_layout.addWidget(QLabel('Input', alignment=Qt.AlignCenter), 0, 0)
        menu_layout.addWidget(QLabel('Method', alignment=Qt.AlignCenter), 0, 1)
        menu_layout.addWidget(QLabel('Degree', alignment=Qt.AlignCenter), 0, 2)
        menu_layout.addLayout(input_grid, 1, 0)
        menu_layout.addLayout(polynomes_grid, 1, 1)
        menu_layout.addLayout(degree_grid_layout, 1, 2)

        self.graphics_tabs = QTabWidget()
        self.graphics_tabs.addTab(pg.PlotWidget(), "Result")

        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(50)
        main_layout.addLayout(menu_layout, 0, 0, 1, -1)

        main_layout.addWidget(self.graphics_tabs, 1, 0)
        main_layout.addWidget(self.text_output, 1, 1)
        main_layout.addWidget(self.__calc_button, 1, 2)

        self.setLayout(main_layout)


    def get_data(self):
        #x1, x2, x3, y = None, None, None, None
        x1_end = self.__x1_dim.value()
        x2_end = x1_end + self.__x2_dim.value()
        x3_end = x2_end + self.__x3_dim.value()
        y_end = x3_end + self.__y_dim.value()
        try:
            data = pd.read_csv(self.__inpuut_file.text())
            x1 = np.array(data.iloc[:, :x1_end])
            x2 = np.array(data.iloc[:, x1_end:x2_end])
            x3 = np.array(data.iloc[:, x2_end:x3_end])
            y = np.array(data.iloc[:, x3_end:y_end])
        except:
            raise Exception('Invalid input info')
        return [x1, x2, x3], y

    def get_method(self):
        method = ''
        for el in [self.__chebyshev, self.__legendre, self.__hermit, self.__laguerre]:
            if el.isChecked():
                method = el.text()
                break
        return method

    def __button_press(self):
        params = {}
        try:
            params['X'], params['y'] = self.get_data()
            params['method'] = self.get_method()
            params['X_degree'] = [self.__x1_degree.value(), self.__x2_degree.value(), self.__x3_degree.value()]
            params['lambda_from_3sys'] = self.__lambda.isChecked()


            res = get_results(params)
            size = res['Y'].shape[0]
            self.graphic_x1.clear()
            self.graphic_x1.plot(range(size), res['Y'][:, 0])
            self.graphic_x1.plot(range(size), res['Y_preds'][:, 0])

            for i in range(self.__y_dim.value()):
                graph = pg.PlotWidget()
                graph.plot(range(size), res['Y'][:, i])
                graph.plot(range(size), res['Y_preds'][:, i])
                self.graphics_tabs.addTab(graph, "Y"+str(i))

                graph = pg.PlotWidget()
                graph.plot(range(size), np.abs(res['Y_err'][:, i]))
                self.graphics_tabs.addTab(graph, "Y" + str(i) + '_error')

            self.text_output.setText(res['logs'])

        except Exception as e:
             print(e)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())



