mkdir data -p
wget https://obs-1253694447.cos.ap-beijing.myqcloud.com/DataSet/Dog_vs_Cat/dogs-vs-cats-redux-kernels-edition.zip -o data/dogs-vs-cats-redux-kernels-edition.zip
mv dogs-vs-cats-redux-kernels-edition.zip ./data
cd data
unzip dogs-vs-cats-redux-kernels-edition.zip
rm dogs-vs-cats-redux-kernels-edition.zip
unzip train.zip
unzip test.zip
