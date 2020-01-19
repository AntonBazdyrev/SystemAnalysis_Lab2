import numpy as np
import pandas as pd
import pyqtgraph as pg
import sys
import qdarkstyle
import time

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QRadioButton, QLabel, QLineEdit, QSpinBox, QCheckBox, QTabWidget, QApplication, QTextBrowser
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot


#from multiplicative_model import get_results
from risk_analysis_model import get_risks_results

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
        self.__output_file = QLineEdit('data/output_own.txt')
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

        self.__relu = QRadioButton('relu', checked=True)
        self.__gelu = QRadioButton('gelu')
        self.__softplus = QRadioButton('softplus')
        self.__sigmoid = QRadioButton('sigmoid')

        self.__calc_button = QPushButton('Calculate Results')
        self.__calc_button.clicked.connect(self.__button_press)

        self.text_output = QTextBrowser()

        self.graphics_tabs = QTabWidget()

        self.progress = QtGui.QProgressBar(self)

        self.__initUI__()

    def __initUI__(self):
        input_grid = QGridLayout()
        input_grid.setVerticalSpacing(5)
        input_grid.addWidget(QLabel('Input file'), 0, 0)
        input_grid.addWidget(QLabel('Output file'), 1, 0)
        #input_grid.addWidget(QLabel('Sample size'), 2, 0)
        input_grid.addWidget(QLabel('X1 dim'), 3, 0)
        input_grid.addWidget(QLabel('X2 dim'), 4, 0)
        input_grid.addWidget(QLabel('X3 dim'), 5, 0)
        input_grid.addWidget(QLabel('Y dim'), 6, 0)
        input_grid.addWidget(self.__inpuut_file, 0, 1)
        input_grid.addWidget(self.__output_file, 1, 1)
        #input_grid.addWidget(self.__sample_size, 2, 1)
        input_grid.addWidget(self.__x1_dim, 3, 1)
        input_grid.addWidget(self.__x2_dim, 4, 1)
        input_grid.addWidget(self.__x3_dim, 5, 1)
        input_grid.addWidget(self.__y_dim, 6, 1)


        polynomes_grid = QGridLayout()
        poly_group = QtGui.QButtonGroup(polynomes_grid)  # poly group
        poly_group.addButton(self.__chebyshev)
        poly_group.addButton(self.__legendre)
        poly_group.addButton(self.__laguerre)
        poly_group.addButton(self.__hermit)
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

        activation_layout = QGridLayout()
        activation_group = QtGui.QButtonGroup(activation_layout)  # activation group
        activation_group.addButton(self.__relu)
        activation_group.addButton(self.__gelu)
        activation_group.addButton(self.__softplus)
        activation_group.addButton(self.__sigmoid)
        activation_layout.addWidget(self.__relu, 0, 0)
        activation_layout.addWidget(self.__gelu, 1, 0)
        activation_layout.addWidget(self.__softplus, 2, 0)
        activation_layout.addWidget(self.__sigmoid, 3, 0)

        menu_layout = QGridLayout()
        menu_layout.setHorizontalSpacing(50)
        menu_layout.addWidget(QLabel('Input', alignment=Qt.AlignCenter), 0, 0)
        menu_layout.addWidget(QLabel('Method', alignment=Qt.AlignCenter), 0, 1)
        menu_layout.addWidget(QLabel('Degree', alignment=Qt.AlignCenter), 0, 2)
        menu_layout.addWidget(QLabel('Activation', alignment=Qt.AlignCenter), 0, 3)
        menu_layout.addLayout(input_grid, 1, 0)
        menu_layout.addLayout(polynomes_grid, 1, 1)
        menu_layout.addLayout(degree_grid_layout, 1, 2)
        menu_layout.addLayout(activation_layout, 1, 3)




        self.graphics_tabs.addTab(pg.PlotWidget(), "Result")
        self.__calc_button.setMaximumWidth(300)

        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(20)
        main_layout.addLayout(menu_layout, 0, 0, 1, -1)

        main_layout.addWidget(self.graphics_tabs, 2, 0)
        main_layout.addWidget(self.text_output, 2, 1)
        main_layout.addWidget(self.progress, 1, 0)
        main_layout.addWidget(self.__calc_button, 1, 1, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

    def get_data(self):
        x1_end = self.__x1_dim.value()
        x2_end = x1_end + self.__x2_dim.value()
        x3_end = x2_end + self.__x3_dim.value()
        y_end = x3_end + self.__y_dim.value()

        data = pd.read_csv(self.__inpuut_file.text())
        x1 = np.array(data.iloc[:, :x1_end])
        x2 = np.array(data.iloc[:, x1_end:x2_end])
        x3 = np.array(data.iloc[:, x2_end:x3_end])
        y = np.array(data.iloc[:, x3_end:y_end])
        return [x1, x2, x3], y

    def get_method(self):
        method = ''
        for el in [self.__chebyshev, self.__legendre, self.__hermit, self.__laguerre]:
            if el.isChecked():
                method = el.text()
        return method

    def get_activation(self):
        method = ''
        for el in [self.__relu, self.__gelu, self.__softplus, self.__sigmoid]:
            if el.isChecked():
                method = el.text()
        return method

    def get_params(self):
        params = {}
        params['X'], params['y'] = self.get_data()
        params['method'] = self.get_method()
        params['X_degree'] = [self.__x1_degree.value(), self.__x2_degree.value(), self.__x3_degree.value()]
        params['lambda_from_3sys'] = self.__lambda.isChecked()
        params['activation'] = self.get_activation()
        return params

    @staticmethod
    def plot_graphs(plot_data, plot_widgets, plot_pens):
        data_dim = len(plot_widgets['Y'])
        for widget_name, widget_data in plot_data.items():
            for dim in range(data_dim):
                for line_name, line_data in widget_data.items():
                    plot_widgets[widget_name][dim].plot(range(len(line_data[dim])),
                                                        line_data[dim],
                                                        pen=plot_pens[widget_name][line_name])

    @pyqtSlot()
    def __button_press(self):
        self.__calc_button.setEnabled(False)
        try:
            params = self.get_params()

            size = params['y'].shape[0]
            dim = params['y'].shape[1]
            plot_data = {'Y': {'y': [[] for _ in range(dim)], 'pred': [[] for _ in range(dim)]},
                         'Y_error': {'error': [[] for _ in range(dim)]},
                         'Y_scaled': {'y': [[] for _ in range(dim)], 'pred': [[] for _ in range(dim)]},
                         'Y_error_scaled': {'error': [[] for _ in range(dim)]}
                         }
            plot_widgets = {'Y': [pg.PlotWidget() for _ in range(dim)],
                            'Y_error': [pg.PlotWidget() for _ in range(dim)],
                            'Y_scaled': [pg.PlotWidget() for _ in range(dim)],
                            'Y_error_scaled': [pg.PlotWidget() for _ in range(dim)]}
            p1 = pg.mkPen(color=(255, 0, 0))
            p2 = pg.mkPen(color=(0, 255, 0))
            p3 = pg.mkPen(color=(0, 0, 255))
            p4 = pg.mkPen(color=(255, 255, 255))
            plot_pens = {'Y': {'y': p1, 'pred': p2},
                         'Y_error': {'error': p4},
                         'Y_scaled': {'y': p1, 'pred': p2},
                         'Y_error_scaled': {'error': p4}
                         }

            self.graphics_tabs.clear()
            for name, val in plot_widgets.items():
                for i, widget in enumerate(val):
                    self.graphics_tabs.addTab(widget, f'{name}{i}')
            QApplication.processEvents()

            for res in get_risks_results(params):
                for i in range(dim):
                    plot_data['Y']['y'][i] = res['Y'][:, i]
                    plot_data['Y']['pred'][i] = res['Y_preds'][:, i]

                    plot_data['Y_scaled']['y'][i] = res['Y_scaled'][:, i]
                    plot_data['Y_scaled']['pred'][i] = res['Y_preds_scaled'][:, i]

                    plot_data['Y_error']['error'][i] = res['Y_err'][:, i]

                    plot_data['Y_error_scaled']['error'][i] = res['Y_err_scaled'][:, i]

                self.plot_graphs(plot_data, plot_widgets, plot_pens)
                QApplication.processEvents()
                time.sleep(0.1)

            self.text_output.setText(res['logs'])

            with open(self.__output_file.text(), 'w') as f:
                f.write(res['logs'])




        except Exception as e:
             print(e)
        self.__calc_button.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())



