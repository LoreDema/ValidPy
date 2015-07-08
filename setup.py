from distutils.core import setup

setup(
    name='ValidPy',
    version='1.0',
    packages=['validpy', 'validpy.ANN', 'validpy.ANN.src', 'validpy.SVM', 'validpy.SVM.src', 'validpy.test',
              'validpy.predict', 'validpy.ANN_vs_SVM', 'validpy.ANN_vs_SVM.src'],
    url='https://github.com/LoreDema/ValidPy',
    license='GPL',
    author='Lorenzo De Mattei',
    author_email='lorenzo.demattei@gmail.com',
    description='Python tool to choose the best configuration and algorithm '
                '(between SVM and ANN) for your machine learning regression task.'
)
