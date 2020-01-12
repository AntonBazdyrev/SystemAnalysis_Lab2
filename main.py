from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QApplication, QRadioButton, QLabel, QLineEdit, QSpinBox, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot, pyqtSignal

import numpy as np
import pandas as pd


def connect(obj, func):
    if isinstance(obj, QSpinBox):
        obj.valueChanged.connect(func)
    else:
        obj.clicked.connect(func)
    return obj


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.__inpuut_file = QLineEdit('data/data_own.csv')
        self.__output_file = QLineEdit('data/output_own.csv')
        self.__sample_size = QSpinBox(value=100)
        self.__x1_dim = QSpinBox(value=7)
        self.__x2_dim = QSpinBox(value=5)
        self.__x3_dim = QSpinBox(value=5)
        self.__y_dim = QSpinBox(value=2)

        self.__chebyshev = QRadioButton('Chebyshev', checked=True)
        self.__legendre = QRadioButton('Legendre')
        self.__hermit = QRadioButton('Hermit')
        self.__laguerre = QRadioButton('Laguerre')

        self.__lambda = QCheckBox('Calculate lambda from 3 systems')

        self.__x1_degree = QSpinBox(value=2)
        self.__x2_degree = QSpinBox(value=3)
        self.__x3_degree = QSpinBox(value=4)

        self.__calc_button = QPushButton('Calculate Results', self)
        self.__calc_button.clicked.connect(self.__button_press)

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

        main_layout = QGridLayout()
        main_layout.addLayout(menu_layout, 0, 0)

        main_layout.addWidget(self.__calc_button, 1, 1)

        self.setLayout(main_layout)

        """
        input_grid = QGridLayout()
        input_grid.setVerticalSpacing(5)
        input_grid.addWidget(QLabel('Input file'), 0, 0)
        input_grid.addWidget(QLabel('Output file'), 1, 0)
        input_grid.addWidget(QLabel('Sample size'), 2, 0)
        input_grid.addWidget(QLabel('X1 dim'), 3, 0)
        input_grid.addWidget(QLabel('X2 dim'), 4, 0)
        input_grid.addWidget(QLabel('X3 dim'), 5, 0)
        input_grid.addWidget(QLabel('Y dim'), 6, 0)
        input_grid.addWidget(QLineEdit(), 0, 1)
        input_grid.addWidget(QLineEdit(), 1, 1)
        input_grid.addWidget(connect(QSpinBox(), self.__print_value), 2, 1)
        input_grid.addWidget(QSpinBox(value=self.__x1_dim), 3, 1)
        input_grid.addWidget(QSpinBox(value=self.__x2_dim), 4, 1)
        input_grid.addWidget(QSpinBox(value=self.__x3_dim), 5, 1)
        input_grid.addWidget(QSpinBox(value=self.__y_dim), 6, 1)

        polynomes_grid = QGridLayout()
        polynomes_grid.addWidget(connect(QRadioButton('Chebyshev'), self.__set_method), 0, 0)
        polynomes_grid.addWidget(connect(QRadioButton('Legendre'), self.__set_method), 1, 0)
        polynomes_grid.addWidget(connect(QRadioButton('Hermit'), self.__set_method), 2, 0)
        polynomes_grid.addWidget(connect(QRadioButton('Laguerre'), self.__set_method), 3, 0)
        polynomes_grid.addWidget(connect(QCheckBox('Визначати lambda з 3 систем рівнянь'), self.__set_lambda), 4, 0)

        degree_grid_layout = QGridLayout()
        degree_grid_layout.addWidget(QLabel('X1'), 0, 0)
        degree_grid_layout.addWidget(QLabel('X2'), 1, 0)
        degree_grid_layout.addWidget(QLabel('X3'), 2, 0)
        degree_grid_layout.addWidget(QSpinBox(value=self.__x1_degree), 0, 1)
        degree_grid_layout.addWidget(QSpinBox(value=self.__x2_degree), 1, 1)
        degree_grid_layout.addWidget(QSpinBox(value=self.__x3_degree), 2, 1)

        menu_layout = QGridLayout()
        menu_layout.setHorizontalSpacing(50)
        menu_layout.addWidget(QLabel('Input', alignment=Qt.AlignCenter), 0, 0)
        menu_layout.addWidget(QLabel('Method', alignment=Qt.AlignCenter), 0, 1)
        menu_layout.addWidget(QLabel('Degree', alignment=Qt.AlignCenter), 0, 2)
        menu_layout.addLayout(input_grid, 1, 0)
        menu_layout.addLayout(polynomes_grid, 1, 1)
        menu_layout.addLayout(degree_grid_layout, 1, 2)

        main_layout = QGridLayout()
        main_layout.addLayout(menu_layout, 0, 0)

        calc_button = QPushButton('Calculate Results', self)
        calc_button.clicked.connect(self.__button_press)
        main_layout.addWidget(calc_button, 1, 1)

        self.main_layout = main_layout
        self.setLayout(main_layout)
        """

        # graphWidget = pg.PlotWidget()
        # hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
        # graphWidget.plot(hour, temperature)
        # grid_layout.addWidget(graphWidget, 0, 0)

    def get_data(self):
        x1, x2, x3, y = None, None, None, None
        x1_end = self.__x1_dim.value() - 1
        x2_end = x1_end + self.__x2_dim.value()
        x3_end = x2_end + self.__x3_dim.value()
        y_end = x3_end + self.__y_dim.value()
        #try:
        data = pd.read_csv(self.__inpuut_file.text())
        x1 = np.array(data.iloc[:, x1_end])
        x2 = np.array(data.iloc[:, x1_end:x2_end])
        x3 = np.array(data.iloc[:, x2_end:x3_end])
        y = np.array(data.iloc[:, x3_end:y_end])
        #except:
        #    print('Invalid input info')
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
        #try:
        params['X'], params['y'] = self.get_data()
        params['method'] = self.get_method()
        params['X_degree'] = [self.__x1_degree.value(), self.__x2_degree.value(), self.__x3_degree.value()]
        params['lambda_from_3sys'] = self.__lambda.isChecked()
       # except Exception as e:
       #     print(e)
        print(params)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())



