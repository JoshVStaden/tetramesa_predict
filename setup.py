from setuptools import setup

setup(
    name='tetramesa_predict',
    version='0.1.0',    
    description='A script for predicting tetramesa',
    url='https://github.com/JoshVStaden/tetramesa_predict.git',
    author='Joshua van Staden',
    author_email='joshvstaden14@gmail.com',
    license='BSD 2-clause',
    packages=['tetramesa_predict'],
    install_requires=['pandas',
                      'numpy',
                      'seaborn',
                      'sklearn',
                      'scikit-image',
                      'keras',
                      'tensorflow-gpu',
                      'keras_tuner'                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)